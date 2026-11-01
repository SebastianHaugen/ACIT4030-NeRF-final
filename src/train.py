"""
Trains the NeRF on the active dataset, adapted from baseline/train_nerf.py (course book Chapter 6).
All settings are read from config.yaml, and the NERF_DATASET / NERF_EXPERIMENT variables pick
the dataset and experiment for one run.

Final project additions:
- sampler.type: uniform (one model, like the baseline) or hierarchical (coarse + fine model, NeRF Sec. 5.2)
- sampler.stratified: random jitter of the sample depths during training
- sampler.density_uses_delta: opacity depends on the sample spacing (see model.py)
"""
import csv
import shutil
import time

import matplotlib
matplotlib.use("Agg")  # saves figures to disk without opening windows
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
)

from src.load_nerf_data import load_nerf_data
from src.model import NeuralRadianceField
from src.render import render_rays
from src.repo_util.LoadConfigurations import (
    CONFIG_PATH,
    ACTIVE_DATASET,
    DATA_DIR,
    OUTPUT_DIR,
    RESIZE_TO,
    CAMERA_ZNEAR,
    CAMERA_ZFAR,
    RAY_MIN_DEPTH,
    RAY_MAX_DEPTH,
    USE_SILHOUETTE_LOSS,
    TEST_EVERY,
    N_RAYS_PER_IMAGE,
    N_PTS_PER_RAY,
    N_ITERATIONS,
    BATCH_SIZE,
    LEARNING_RATE,
    LR_DECAY_AT,
    LR_DECAY_FACTOR,
    HUBER_SCALING,
    RANDOM_SEED,
    VISUALIZE_EVERY,
    CHECKPOINT_EVERY,
    SAMPLER,
    STRATIFIED,
    N_COARSE,
    N_FINE,
    DENSITY_USES_DELTA,
    EXPERIMENT,
)

PREVIEW_VIEW = 0  # the same training view in every preview, so they can be compared over time
HIERARCHICAL = SAMPLER == "hierarchical"
if SAMPLER not in ("uniform", "hierarchical"):
    raise ValueError(f"Unknown sampler.type '{SAMPLER}', expected 'uniform' or 'hierarchical'")

# Uniform uses n_pts_per_ray samples, hierarchical uses n_coarse in the first pass (+ n_fine in the second)
N_PTS_FIRST_PASS = N_COARSE if HIERARCHICAL else N_PTS_PER_RAY


# Helper functions, huber and sample_images_at_mc_locs come from the baseline
def huber(x, y, scaling=HUBER_SCALING):
    """Smooth L1 (Huber) loss between the rendered and the real colours or silhouettes."""
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss


def sample_images_at_mc_locs(target_images, sampled_rays_xy, ndc_extent=(1.0, 1.0)):
    """Reads the real pixel colours at the random ray positions used in training."""
    # ndc_extent is our addition for non-square images, square images give the baseline behaviour
    ba = target_images.shape[0]
    dim = target_images.shape[-1]
    spatial_size = sampled_rays_xy.shape[1:-1]
    extent = torch.tensor(ndc_extent, device=sampled_rays_xy.device)
    # The sign is flipped because PyTorch3D and grid_sample use opposite x and y directions
    grid = -(sampled_rays_xy / extent)
    images_sampled = torch.nn.functional.grid_sample(
        target_images.permute(0, 3, 1, 2),
        grid.view(ba, -1, 1, 2),
        align_corners=True,
    )
    return images_sampled.permute(0, 2, 3, 1).view(ba, *spatial_size, dim)


def ndc_extent_for(height, width):
    """Returns how far the NDC coordinates reach in x and y for this image size."""
    # In PyTorch3D the shorter side reaches 1 and the longer side reaches long / short
    if width >= height:
        return (width / height, 1.0)
    return (1.0, height / width)


def fix_fov_for_aspect(cameras, height, width):
    """Converts the field of view so portrait images such as poster are rendered correctly."""
    # PyTorch3D applies the field of view to the shorter side, but load_nerf_data gives the vertical one
    # Square and landscape images such as lego need no change
    if width >= height:
        return cameras
    # Converts the vertical field of view into the horizontal one
    fov_y = torch.deg2rad(cameras.fov)
    fov_short = 2 * torch.atan(torch.tan(fov_y / 2) * width / height)
    return FoVPerspectiveCameras(
        R=cameras.R,
        T=cameras.T,
        fov=torch.rad2deg(fov_short),
        znear=cameras.znear,
        zfar=cameras.zfar,
        device=cameras.device,
    )


def select_cameras(cameras, idx):
    """Builds a camera batch from the given indices, the same way as the baseline."""
    return FoVPerspectiveCameras(
        R=cameras.R[idx],
        T=cameras.T[idx],
        znear=cameras.znear[idx],
        zfar=cameras.zfar[idx],
        aspect_ratio=cameras.aspect_ratio[idx],
        fov=cameras.fov[idx],
        device=cameras.device,
    )


def load_train_split(device):
    """Loads the training images, silhouettes and cameras of the active dataset."""
    images, silhouettes, cameras = load_nerf_data(
        str(DATA_DIR),
        split="train",
        device=device,
        znear=CAMERA_ZNEAR,
        zfar=CAMERA_ZFAR,
        resize_to=RESIZE_TO,
    )
    # Bicubic resizing can give values slightly below 0 or above 1
    images = images.clamp(0.0, 1.0)
    silhouettes = silhouettes.clamp(0.0, 1.0)

    # Poster has no test split, so every 8th image (TEST_EVERY) is held out here
    if TEST_EVERY:
        all_idx = torch.arange(len(images))
        train_idx = all_idx[all_idx % TEST_EVERY != 0]
        images = images[train_idx]
        silhouettes = silhouettes[train_idx]
        cameras = select_cameras(cameras, train_idx.to(device))
        print(f"Held out every {TEST_EVERY}th image: {len(train_idx)} train images left.")

    height, width = images.shape[1:3]
    cameras = fix_fov_for_aspect(cameras, height, width)
    return images, silhouettes, cameras


def make_train_raysampler(x_extent, y_extent):
    """Random rays for each training step, optionally with stratified (jittered) sample depths."""
    kwargs = dict(
        min_x=-x_extent,
        max_x=x_extent,
        min_y=-y_extent,
        max_y=y_extent,
        n_rays_per_image=N_RAYS_PER_IMAGE,
        n_pts_per_ray=N_PTS_FIRST_PASS,
        min_depth=RAY_MIN_DEPTH,
        max_depth=RAY_MAX_DEPTH,
    )
    if STRATIFIED:
        kwargs["stratified_sampling"] = True
    try:
        return MonteCarloRaysampler(**kwargs)
    except TypeError as e:
        raise RuntimeError(
            "This PyTorch3D version does not support stratified_sampling in MonteCarloRaysampler "
            "(added in 0.7.x). Update PyTorch3D or set sampler.stratified: false."
        ) from e


def compute_losses(rendered, xys, target_images, target_silhouettes, batch_idx, extent):
    """Colour loss (and silhouette loss for lego) for one rendered ray batch."""
    rendered_images, rendered_silhouettes = rendered.split([3, 1], dim=-1)

    # Real colours at the same pixels as the rendered rays
    colors_at_rays = sample_images_at_mc_locs(target_images[batch_idx], xys, extent)
    color_err = huber(rendered_images, colors_at_rays).abs().mean()

    sil_err = None
    # Only lego has an alpha channel for the silhouette loss
    if USE_SILHOUETTE_LOSS:
        silhouettes_at_rays = sample_images_at_mc_locs(
            target_silhouettes[batch_idx, ..., None], xys, extent
        )
        sil_err = huber(rendered_silhouettes, silhouettes_at_rays).abs().mean()
    return color_err, sil_err


def save_preview(coarse, fine, raysampler_grid, raymarcher, camera, target_image,
                 hist_color, hist_sil, iteration, path):
    """Saves a full render of one training view next to the ground truth and the loss curves."""
    with torch.no_grad():
        ray_bundle = raysampler_grid(cameras=camera)
        rendered, rendered_coarse = render_rays(
            ray_bundle, coarse, fine, raymarcher, N_FINE, training=False, chunked=True
        )
    rendered_image = rendered[0, ..., :3].clamp(0.0, 1.0)
    target_image = target_image.clamp(0.0, 1.0)
    # Quick PSNR for the preview title
    psnr = -10 * torch.log10(((rendered_image - target_image) ** 2).mean())

    # Hierarchical runs also show the coarse render, so you can see what the fine pass adds
    n_img = 3 if rendered_coarse is not None else 2
    fig, ax = plt.subplots(1, n_img + 1, figsize=(5 * (n_img + 1), 5))
    ax[0].imshow(rendered_image.cpu().numpy())
    ax[0].set_title(f"{'Fine' if fine is not None else 'Render'} (iter {iteration}, PSNR {psnr:.2f} dB)")
    col = 1
    if rendered_coarse is not None:
        coarse_image = rendered_coarse[0, ..., :3].clamp(0.0, 1.0)
        psnr_c = -10 * torch.log10(((coarse_image - target_image) ** 2).mean())
        ax[col].imshow(coarse_image.cpu().numpy())
        ax[col].set_title(f"Coarse (PSNR {psnr_c:.2f} dB)")
        col += 1
    ax[col].imshow(target_image.cpu().numpy())
    ax[col].set_title("Ground truth")
    ax[-1].plot(hist_color, label="colour")
    if hist_sil:
        ax[-1].plot(hist_sil, label="silhouette")
    ax[-1].set_yscale("log")
    ax[-1].set_title("Training loss")
    ax[-1].legend()
    for a in ax[:-1]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_checkpoint(coarse, fine, iteration, loss_history, path):
    """Saves the model weights with the dataset and sampler settings, so evaluate.py can check them."""
    torch.save(
        {
            "iteration": iteration,
            "dataset": ACTIVE_DATASET,
            "experiment": EXPERIMENT,
            "sampler": {
                "type": SAMPLER,
                "stratified": STRATIFIED,
                "n_pts_first_pass": N_PTS_FIRST_PASS,
                "n_fine": N_FINE if HIERARCHICAL else 0,
                "density_uses_delta": DENSITY_USES_DELTA,
            },
            "coarse_state_dict": coarse.state_dict(),
            "fine_state_dict": fine.state_dict() if fine is not None else None,
            "loss_history": loss_history,
        },
        path,
    )


def save_loss_history_csv(color, silhouette, coarse_color, total, seconds, path):
    """Saves the losses and the elapsed training time with one row per iteration."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "color_loss", "silhouette_loss", "coarse_color_loss",
                         "total_loss", "elapsed_s"])
        for i, (c, t, s_el) in enumerate(zip(color, total, seconds)):
            s = silhouette[i] if i < len(silhouette) else ""        # empty for poster
            cc = coarse_color[i] if i < len(coarse_color) else ""   # empty for uniform sampling
            writer.writerow([i, c, s, cc, t, f"{s_el:.2f}"])


# Training
def main(
    n_iter=N_ITERATIONS,
    output_dir=OUTPUT_DIR,
    vis_every=VISUALIZE_EVERY,
    checkpoint_every=CHECKPOINT_EVERY,
):
    """Trains on the active dataset and returns the loss histories."""
    # The arguments come from config.yaml, only smoke_test.py passes other values
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(RANDOM_SEED)

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    # Saves a copy of the settings this run used, the copy is never read as a config
    config_copy = output_dir / "config_used.yaml"
    shutil.copy(CONFIG_PATH, config_copy)
    with open(config_copy, "a", encoding="utf-8") as f:
        f.write(f"\n# This run: dataset = {ACTIVE_DATASET}, experiment = {EXPERIMENT}, n_iter = {n_iter}\n")

    # Data
    print(f"Dataset: {ACTIVE_DATASET}  ({DATA_DIR})")
    print(f"Experiment: {EXPERIMENT}  sampler={SAMPLER}, stratified={STRATIFIED}, "
          f"density_uses_delta={DENSITY_USES_DELTA}, points per ray={N_PTS_FIRST_PASS}"
          + (f" + {N_FINE} fine" if HIERARCHICAL else ""))
    target_images, target_silhouettes, target_cameras = load_train_split(device)
    n_images, height, width, _ = target_images.shape
    x_extent, y_extent = ndc_extent_for(height, width)
    print(f"{n_images} training images at {width}x{height}, {n_iter} iterations on {device}")

    # Ray samplers and raymarcher
    raymarcher = EmissionAbsorptionRaymarcher()
    raysampler_mc = make_train_raysampler(x_extent, y_extent)
    # One ray per pixel for the full image previews, never jittered
    raysampler_grid = NDCMultinomialRaysampler(
        image_height=height,
        image_width=width,
        n_pts_per_ray=N_PTS_FIRST_PASS,
        min_depth=RAY_MIN_DEPTH,
        max_depth=RAY_MAX_DEPTH,
    )

    # Models and optimizer. Hierarchical sampling trains a separate fine model, as in NeRF
    coarse = NeuralRadianceField().to(device)
    fine = NeuralRadianceField().to(device) if HIERARCHICAL else None
    params = list(coarse.parameters()) + (list(fine.parameters()) if fine is not None else [])
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    lr_decay_iteration = round(n_iter * LR_DECAY_AT)  # iteration where the learning rate drops

    loss_history_color, loss_history_sil, loss_history_coarse, loss_history_total = [], [], [], []
    elapsed = []              # cumulative training time per iteration, for the wall-clock plots
    train_seconds = 0.0       # counts only the training steps, not previews or checkpoints
    start_time = time.time()
    extent = (x_extent, y_extent)

    for iteration in tqdm(range(n_iter), desc="Training"):
        step_start = time.time()
        if iteration == lr_decay_iteration:
            # Same as the baseline, a new optimizer with a lower learning rate
            tqdm.write(f"Decreasing LR by factor {LR_DECAY_FACTOR} ...")
            optimizer = torch.optim.Adam(params, lr=LEARNING_RATE * LR_DECAY_FACTOR)

        optimizer.zero_grad()
        # Picks BATCH_SIZE random training images
        batch_idx = torch.randperm(n_images)[:BATCH_SIZE]
        batch_cameras = select_cameras(target_cameras, batch_idx)

        # Random rays, then the coarse pass (and the fine pass if hierarchical)
        ray_bundle = raysampler_mc(cameras=batch_cameras)
        rendered, rendered_coarse = render_rays(
            ray_bundle, coarse, fine, raymarcher, N_FINE, training=True
        )

        # Loss on the final render (the fine render if hierarchical)
        color_err, sil_err = compute_losses(
            rendered, ray_bundle.xys, target_images, target_silhouettes, batch_idx, extent
        )
        loss = color_err + (sil_err if sil_err is not None else 0.0)

        # NeRF also trains the coarse model on the same targets, so its weights stay useful
        if rendered_coarse is not None:
            color_err_c, sil_err_c = compute_losses(
                rendered_coarse, ray_bundle.xys, target_images, target_silhouettes, batch_idx, extent
            )
            loss = loss + color_err_c + (sil_err_c if sil_err_c is not None else 0.0)
            loss_history_coarse.append(color_err_c.item())

        if sil_err is not None:
            loss_history_sil.append(sil_err.item())
        loss_history_color.append(color_err.item())
        loss_history_total.append(loss.item())

        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()  # waits for the GPU to finish, so the step time is accurate
        train_seconds += time.time() - step_start
        elapsed.append(train_seconds)

        # Saves a preview regularly and at the last iteration
        if iteration % vis_every == 0 or iteration == n_iter - 1:
            save_preview(
                coarse,
                fine,
                raysampler_grid,
                raymarcher,
                select_cameras(target_cameras, [PREVIEW_VIEW]),
                target_images[PREVIEW_VIEW],
                loss_history_color,
                loss_history_sil,
                iteration,
                preview_dir / f"iter_{iteration:05d}.png",
            )

        if iteration > 0 and iteration % checkpoint_every == 0:
            save_checkpoint(coarse, fine, iteration, loss_history_total, output_dir / "checkpoint.pt")

    # Final checkpoint and loss history for the report
    save_checkpoint(coarse, fine, n_iter, loss_history_total, output_dir / "checkpoint.pt")
    save_loss_history_csv(
        loss_history_color,
        loss_history_sil,
        loss_history_coarse,
        loss_history_total,
        elapsed,
        output_dir / "loss_history.csv",
    )
    total_time = time.time() - start_time
    print(f"Done in {total_time / 60:.1f} min total, {train_seconds / 60:.1f} min training "
          f"({train_seconds / n_iter:.3f} s/iteration, previews and checkpoints excluded). "
          f"Outputs in {output_dir}")
    return {"color": loss_history_color, "silhouette": loss_history_sil, "total": loss_history_total}


if __name__ == "__main__":
    main()