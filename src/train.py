"""
Trains the NeRF on the active dataset, adapted from baseline/train_nerf.py (course book Chapter 6).
All settings are read from config.yaml, and the NERF_DATASET / NERF_EXPERIMENT variables pick
the dataset and experiment for one run.

Final project additions:
- sampler.type: uniform (one model, like the baseline), hierarchical (coarse + fine model, NeRF Sec. 5.2)
  or proposal (small proposal network + NeRF model, mip-NeRF 360 Sec. 3)
- sampler.stratified: random jitter of the sample depths during training
- sampler.density_uses_delta: opacity depends on the sample spacing (see model.py)
- sampler.spacing / sampler.contract: lin-disp spacing and scene contraction (mip-NeRF 360 Sec. 2)
- model.*: NeRF positional encoding and 8-layer MLP, per-image appearance codes (NeRF-W)
- train.distortion_weight: distortion loss (mip-NeRF 360 Sec. 4)
"""
import csv
import math
import time
from dataclasses import asdict

import yaml

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

from src.load_nerf_data import load_nerf_data, load_frame_times
from src.model import NeuralRadianceField, ProposalField
from src.render import render_rays, SamplerSettings, SAMPLERS
from src.sampling import interlevel_loss, distortion_loss, lengths_to_edges, t_to_s, SPACINGS
from src.repo_util.LoadConfigurations import (
    CONFIG,
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
    N_HARMONIC_FUNCTIONS,
    N_HARMONIC_FUNCTIONS_DIR,
    HARMONIC_OMEGA0,
    INCLUDE_INPUT,
    N_HIDDEN_NEURONS,
    ARCHITECTURE,
    APPEARANCE_DIM,
    SPACING,
    CONTRACT,
    CONTRACTION_CENTER,
    CONTRACTION_RADIUS,
    LINDISP_SWITCH,
    PROPOSAL_HIDDEN,
    DISTORTION_WEIGHT,
    INTERLEVEL_WEIGHT,
    LR_SCHEDULE,
    LR_FINAL,
    LR_WARMUP,
    FIX_DATA,
    SCENE_BOUND,
    HASHGRID,
    ADAM_EPS,
    TIME_INPUT,
    N_HARMONIC_FUNCTIONS_TIME,
)

PREVIEW_VIEW = 0  # the same training view in every preview, so they can be compared over time
if SAMPLER not in SAMPLERS:
    raise ValueError(f"Unknown sampler.type '{SAMPLER}', expected one of {SAMPLERS}")
if SPACING not in SPACINGS:
    raise ValueError(f"Unknown sampler.spacing '{SPACING}', expected one of {SPACINGS}")
if (SAMPLER == "proposal" or SPACING != "linear") and not DENSITY_USES_DELTA:
    raise ValueError("Proposal sampling and non-linear spacing give unevenly spaced samples, "
                     "so they need sampler.density_uses_delta: true")
if LR_SCHEDULE not in ("step", "exp"):
    raise ValueError(f"Unknown train.lr_schedule '{LR_SCHEDULE}', expected 'step' or 'exp'")
if ARCHITECTURE == "hashgrid" and not DENSITY_USES_DELTA:
    raise ValueError("architecture: hashgrid predicts a density per unit length, so it needs "
                     "sampler.density_uses_delta: true")

# Uniform uses n_pts_per_ray samples, the two-pass samplers use n_coarse in the first pass
TWO_PASS = SAMPLER != "uniform"
N_PTS_FIRST_PASS = N_COARSE if TWO_PASS else N_PTS_PER_RAY

SAMPLER_SETTINGS = SamplerSettings(
    type=SAMPLER,
    n_first=N_PTS_FIRST_PASS,
    n_fine=N_FINE if TWO_PASS else 0,
    min_depth=RAY_MIN_DEPTH,
    max_depth=RAY_MAX_DEPTH,
    spacing=SPACING,
    lindisp_switch=LINDISP_SWITCH,
    stratified=STRATIFIED,
)

# Saved in the checkpoint, so evaluate.py rebuilds exactly the same networks
NERF_KWARGS = dict(
    n_harmonic_functions=N_HARMONIC_FUNCTIONS,
    n_hidden_neurons=N_HIDDEN_NEURONS,
    omega0=HARMONIC_OMEGA0,
    density_uses_delta=DENSITY_USES_DELTA,
    n_harmonic_functions_dir=N_HARMONIC_FUNCTIONS_DIR,
    include_input=INCLUDE_INPUT,
    architecture=ARCHITECTURE,
    appearance_dim=APPEARANCE_DIM,
    contract=CONTRACT,
    contraction_center=CONTRACTION_CENTER,
    contraction_radius=CONTRACTION_RADIUS,
    scene_bound=SCENE_BOUND,
    hashgrid=HASHGRID,
    time_input=TIME_INPUT,
    n_harmonic_functions_time=N_HARMONIC_FUNCTIONS_TIME,
)
PROPOSAL_KWARGS = dict(
    n_harmonic_functions=N_HARMONIC_FUNCTIONS,
    omega0=HARMONIC_OMEGA0,
    include_input=INCLUDE_INPUT,
    n_hidden_neurons=PROPOSAL_HIDDEN,
    contract=CONTRACT,
    contraction_center=CONTRACTION_CENTER,
    contraction_radius=CONTRACTION_RADIUS,
    # A hash-grid NeRF gets a small hash-grid proposal network, as in nerfacto
    architecture="hashgrid" if ARCHITECTURE == "hashgrid" else "mlp",
    scene_bound=SCENE_BOUND,
)


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
        premultiply_alpha=FIX_DATA,
        undistort=FIX_DATA,
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


def load_train_times(device):
    """Frame times (n_train,) of the training images for time_input models, else None."""
    if not TIME_INPUT:
        return None
    times = load_frame_times(str(DATA_DIR), split="train")
    if times is None:
        raise ValueError(f"model.time_input is on, but the {ACTIVE_DATASET} frames have no 'time' field")
    if TEST_EVERY:  # the same images as load_train_split keeps
        all_idx = torch.arange(len(times))
        times = times[all_idx % TEST_EVERY != 0]
    return times.to(device)


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
    # With a non-linear spacing, render_rays replaces the depths and does the jitter itself
    if STRATIFIED and SPACING == "linear":
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


def exp_learning_rate(iteration, n_iter):
    """Smooth learning-rate schedule (train.lr_schedule: exp), as in NeRF and mip-NeRF 360.

    Decays log-linearly from learning_rate to lr_final over the run, after an optional linear
    warm-up over the first lr_warmup iterations, which keeps the deep MLP stable at the start.
    """
    progress = iteration / max(1, n_iter - 1)
    lr = math.exp((1 - progress) * math.log(LEARNING_RATE) + progress * math.log(LR_FINAL))
    if LR_WARMUP > 0:
        lr *= min(1.0, (iteration + 1) / LR_WARMUP)
    return lr


def ray_distortion_loss(out, settings):
    """Distortion loss on the final weights, measured in the normalised distance s (mip-NeRF 360)."""
    s_edges = t_to_s(
        lengths_to_edges(out["lengths"].detach()),
        settings.spacing, settings.min_depth, settings.max_depth, settings.lindisp_switch,
    )
    return distortion_loss(s_edges, out["weights"])


def build_models(settings, nerf_kwargs, proposal_kwargs, n_images, device):
    """Creates the networks for the sampler type.

    Returns (first, second, appearance):
        uniform:      NeRF, None, ...
        hierarchical: coarse NeRF, fine NeRF, ...
        proposal:     ProposalField, NeRF, ...
    appearance is an embedding with one code per training image, or None if appearance_dim is 0.
    """
    if settings.type == "proposal":
        first = ProposalField(**proposal_kwargs).to(device)
    else:
        first = NeuralRadianceField(**nerf_kwargs).to(device)
    second = NeuralRadianceField(**nerf_kwargs).to(device) if settings.type != "uniform" else None
    appearance = None
    if nerf_kwargs.get("appearance_dim", 0) > 0:
        appearance = torch.nn.Embedding(n_images, nerf_kwargs["appearance_dim"]).to(device)
    return first, second, appearance


def save_preview(first, second, appearance, raysampler_grid, raymarcher, camera, target_image,
                 hist_color, hist_sil, iteration, path, train_times=None):
    """Saves a full render of one training view next to the ground truth and the loss curves."""
    with torch.no_grad():
        ray_bundle = raysampler_grid(cameras=camera)
        codes = None
        if appearance is not None:
            codes = appearance(torch.tensor([PREVIEW_VIEW], device=target_image.device))
        out = render_rays(
            ray_bundle, first, second, raymarcher, SAMPLER_SETTINGS, training=False, chunked=True,
            appearance_codes=codes,
            times=train_times[PREVIEW_VIEW:PREVIEW_VIEW + 1] if train_times is not None else None,
        )
    rendered, rendered_coarse = out["rgba"], out["rgba_coarse"]
    rendered_image = rendered[0, ..., :3].clamp(0.0, 1.0)
    target_image = target_image.clamp(0.0, 1.0)
    # Quick PSNR for the preview title
    psnr = -10 * torch.log10(((rendered_image - target_image) ** 2).mean())

    # Hierarchical runs also show the coarse render, so you can see what the fine pass adds
    n_img = 3 if rendered_coarse is not None else 2
    fig, ax = plt.subplots(1, n_img + 1, figsize=(5 * (n_img + 1), 5))
    ax[0].imshow(rendered_image.cpu().numpy())
    ax[0].set_title(f"{'Fine' if second is not None else 'Render'} (iter {iteration}, PSNR {psnr:.2f} dB)")
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


def save_checkpoint(first, second, appearance, iteration, loss_history, path):
    """Saves the model weights with the dataset, sampler and model settings, so evaluate.py can
    rebuild exactly the same networks."""
    proposal = SAMPLER == "proposal"
    torch.save(
        {
            "iteration": iteration,
            "dataset": ACTIVE_DATASET,
            "experiment": EXPERIMENT,
            "fix_data": FIX_DATA,
            "sampler": asdict(SAMPLER_SETTINGS),
            "nerf_kwargs": NERF_KWARGS,
            "proposal_kwargs": PROPOSAL_KWARGS if proposal else None,
            # uniform: the only model, hierarchical: the coarse model
            "coarse_state_dict": first.state_dict() if not proposal else None,
            "proposal_state_dict": first.state_dict() if proposal else None,
            # hierarchical: the fine model, proposal: the NeRF model
            "fine_state_dict": second.state_dict() if second is not None else None,
            "appearance_state_dict": appearance.state_dict() if appearance is not None else None,
            "loss_history": loss_history,
        },
        path,
    )


def save_loss_history_csv(color, silhouette, coarse_color, interlevel, distortion, total, seconds, path):
    """Saves the losses and the elapsed training time with one row per iteration."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "color_loss", "silhouette_loss", "coarse_color_loss",
                         "total_loss", "elapsed_s", "interlevel_loss", "distortion_loss"])
        for i, (c, t, s_el) in enumerate(zip(color, total, seconds)):
            s = silhouette[i] if i < len(silhouette) else ""        # empty for poster
            cc = coarse_color[i] if i < len(coarse_color) else ""   # empty unless hierarchical
            il = interlevel[i] if i < len(interlevel) else ""       # empty unless proposal
            d = distortion[i] if i < len(distortion) else ""        # empty if the distortion loss is off
            writer.writerow([i, c, s, cc, t, f"{s_el:.2f}", il, d])


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
    # Saves the settings this run actually used, including any NERF_* environment overrides,
    # so report_figures.py and the report always see the real settings. Never read as a config
    config_copy = output_dir / "config_used.yaml"
    with open(config_copy, "w", encoding="utf-8") as f:
        yaml.safe_dump(CONFIG, f, sort_keys=False)
        f.write(f"\n# This run: dataset = {ACTIVE_DATASET}, experiment = {EXPERIMENT}, n_iter = {n_iter}\n")

    # Data
    print(f"Dataset: {ACTIVE_DATASET}  ({DATA_DIR}), data fixes: {'on' if FIX_DATA else 'off'}")
    print(f"Experiment: {EXPERIMENT}  sampler={SAMPLER}, stratified={STRATIFIED}, "
          f"density_uses_delta={DENSITY_USES_DELTA}, points per ray={N_PTS_FIRST_PASS}"
          + (f" + {N_FINE} fine" if SAMPLER == "hierarchical" else "")
          + (f" -> {N_FINE} NeRF samples" if SAMPLER == "proposal" else "")
          + f", spacing={SPACING}, contract={CONTRACT}")
    print(f"Model: {ARCHITECTURE}, L={N_HARMONIC_FUNCTIONS}/{N_HARMONIC_FUNCTIONS_DIR}, omega0={HARMONIC_OMEGA0}, "
          f"include_input={INCLUDE_INPUT}, appearance_dim={APPEARANCE_DIM}, lr={LEARNING_RATE} ({LR_SCHEDULE}"
          + (f" -> {LR_FINAL}, warm-up {LR_WARMUP}" if LR_SCHEDULE == "exp" else "")
          + f"), distortion_weight={DISTORTION_WEIGHT}, rays per step={BATCH_SIZE}x{N_RAYS_PER_IMAGE}"
          + (f", time input (L={N_HARMONIC_FUNCTIONS_TIME})" if TIME_INPUT else ""))
    target_images, target_silhouettes, target_cameras = load_train_split(device)
    train_times = load_train_times(device)
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

    # Models and optimizer. Hierarchical sampling trains a separate fine model, as in NeRF,
    # proposal sampling trains a small proposal network next to the NeRF model
    first, second, appearance = build_models(SAMPLER_SETTINGS, NERF_KWARGS, PROPOSAL_KWARGS, n_images, device)
    params = [p for m in (first, second, appearance) if m is not None for p in m.parameters()]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, eps=ADAM_EPS)
    lr_decay_iteration = round(n_iter * LR_DECAY_AT)  # iteration where the learning rate drops

    loss_history_color, loss_history_sil, loss_history_coarse, loss_history_total = [], [], [], []
    loss_history_interlevel, loss_history_dist = [], []
    elapsed = []              # cumulative training time per iteration, for the wall-clock plots
    train_seconds = 0.0       # counts only the training steps, not previews or checkpoints
    start_time = time.time()
    extent = (x_extent, y_extent)

    for iteration in tqdm(range(n_iter), desc="Training"):
        step_start = time.time()
        if LR_SCHEDULE == "exp":
            for group in optimizer.param_groups:
                group["lr"] = exp_learning_rate(iteration, n_iter)
        elif iteration == lr_decay_iteration:
            # Same as the baseline, a new optimizer with a lower learning rate
            tqdm.write(f"Decreasing LR by factor {LR_DECAY_FACTOR} ...")
            optimizer = torch.optim.Adam(params, lr=LEARNING_RATE * LR_DECAY_FACTOR, eps=ADAM_EPS)

        optimizer.zero_grad()
        # Picks BATCH_SIZE random training images
        batch_idx = torch.randperm(n_images)[:BATCH_SIZE]
        batch_cameras = select_cameras(target_cameras, batch_idx)
        # The appearance code of each image in the batch (NeRF-W)
        codes = appearance(batch_idx.to(device)) if appearance is not None else None
        # The time of each image in the batch (dynamic scenes, T-NeRF)
        batch_times = train_times[batch_idx.to(device)] if train_times is not None else None

        # Random rays, then the first pass (and the second pass for hierarchical / proposal)
        ray_bundle = raysampler_mc(cameras=batch_cameras)
        out = render_rays(
            ray_bundle, first, second, raymarcher, SAMPLER_SETTINGS, training=True, appearance_codes=codes,
            times=batch_times,
        )

        # Loss on the final render (the fine / NeRF render for the two-pass samplers)
        color_err, sil_err = compute_losses(
            out["rgba"], ray_bundle.xys, target_images, target_silhouettes, batch_idx, extent
        )
        loss = color_err + (sil_err if sil_err is not None else 0.0)

        # NeRF also trains the coarse model on the same targets, so its weights stay useful
        if out["rgba_coarse"] is not None:
            color_err_c, sil_err_c = compute_losses(
                out["rgba_coarse"], ray_bundle.xys, target_images, target_silhouettes, batch_idx, extent
            )
            loss = loss + color_err_c + (sil_err_c if sil_err_c is not None else 0.0)
            loss_history_coarse.append(color_err_c.item())

        # The proposal network only learns to cover the NeRF weights (mip-NeRF 360, eq. 14)
        if out["proposal_weights"] is not None:
            inter_err = interlevel_loss(
                out["lengths"], out["weights"], out["proposal_lengths"], out["proposal_weights"]
            )
            loss = loss + INTERLEVEL_WEIGHT * inter_err
            loss_history_interlevel.append(inter_err.item())

        # Pulls the weights on each ray together, against floaters (mip-NeRF 360, eq. 15)
        if DISTORTION_WEIGHT > 0:
            dist_err = ray_distortion_loss(out, SAMPLER_SETTINGS)
            loss = loss + DISTORTION_WEIGHT * dist_err
            loss_history_dist.append(dist_err.item())

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
                first,
                second,
                appearance,
                raysampler_grid,
                raymarcher,
                select_cameras(target_cameras, [PREVIEW_VIEW]),
                target_images[PREVIEW_VIEW],
                loss_history_color,
                loss_history_sil,
                iteration,
                preview_dir / f"iter_{iteration:05d}.png",
                train_times,
            )

        if iteration > 0 and iteration % checkpoint_every == 0:
            save_checkpoint(first, second, appearance, iteration, loss_history_total,
                            output_dir / "checkpoint.pt")

    # Final checkpoint and loss history for the report
    save_checkpoint(first, second, appearance, n_iter, loss_history_total, output_dir / "checkpoint.pt")
    save_loss_history_csv(
        loss_history_color,
        loss_history_sil,
        loss_history_coarse,
        loss_history_interlevel,
        loss_history_dist,
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