"""
Evaluates a trained NeRF on the held-out test views of the active dataset.
Run from the project root after training with python -m src.evaluate
Works for uniform, hierarchical and proposal checkpoints. The networks and sampler settings are
rebuilt from the checkpoint, so evaluation always renders the same way as training did.
"""
import csv
import json
import os
import re
import statistics

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pytorch3d.renderer import (
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
)

from src.load_nerf_data import load_nerf_data
from src.metrics import psnr, ssim, LPIPSMetric
from src.render import render_rays, SamplerSettings
from src.train import select_cameras, fix_fov_for_aspect, build_models
from src.repo_util.LoadConfigurations import (
    ACTIVE_DATASET,
    DATA_DIR,
    OUTPUT_DIR,
    FIGURES_DIR,
    RESIZE_TO,
    CAMERA_ZNEAR,
    CAMERA_ZFAR,
    RAY_MIN_DEPTH,
    RAY_MAX_DEPTH,
    DENSITY_USES_DELTA,
    TEST_EVERY,
    MAX_TEST_VIEWS,
    N_FIGURE_VIEWS,
    EXPERIMENT,
    FIX_DATA,
    EVAL_APPEARANCE,
)

IEEE_TEXT_WIDTH_IN = 7.16    # full page width of an IEEE two-column paper in inches
MAX_FIGURE_HEIGHT_IN = 5.0   # keeps the figures well below the page height


def load_test_split(device):
    """Loads the test images and cameras of the active dataset."""
    # Poster has no test split, so it uses the same images that train.py held out (every 8th by default)
    if TEST_EVERY:
        images, _, cameras = load_nerf_data(
            str(DATA_DIR), split="train", device=device,
            znear=CAMERA_ZNEAR, zfar=CAMERA_ZFAR, resize_to=RESIZE_TO,
            premultiply_alpha=FIX_DATA, undistort=FIX_DATA,
        )
        all_idx = torch.arange(len(images))
        test_idx = all_idx[all_idx % TEST_EVERY == 0]
    # Lego comes with its own test split
    else:
        images, _, cameras = load_nerf_data(
            str(DATA_DIR), split="test", device=device,
            znear=CAMERA_ZNEAR, zfar=CAMERA_ZFAR, resize_to=RESIZE_TO,
            premultiply_alpha=FIX_DATA, undistort=FIX_DATA,
        )
        test_idx = torch.arange(len(images))

    if MAX_TEST_VIEWS and len(test_idx) > MAX_TEST_VIEWS:
        # Evenly spaced subset to save time on large test splits
        pick = torch.linspace(0, len(test_idx) - 1, MAX_TEST_VIEWS).round().long().unique()
        test_idx = test_idx[pick]

    images = images[test_idx].clamp(0.0, 1.0)
    cameras = select_cameras(cameras, test_idx.to(device))
    # Same field of view correction as in training, otherwise portrait views would not line up
    cameras = fix_fov_for_aspect(cameras, *images.shape[1:3])
    return images, cameras, test_idx.tolist()


def load_model(device):
    """Loads the networks and the sampler settings of the trained run.

    Returns (first, second, appearance_weights, settings), see build_models in train.py.
    appearance_weights holds one appearance code per training image (n_train, dim), or None.
    """
    checkpoint_path = OUTPUT_DIR / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}. Train first: python -m src.train")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Stops a lego model from being evaluated on poster by mistake, and the other way round
    if checkpoint.get("dataset") != ACTIVE_DATASET:
        raise ValueError(
            f"Checkpoint was trained on '{checkpoint.get('dataset')}', "
            f"but active_dataset is '{ACTIVE_DATASET}'."
        )
    # The test images must be loaded the same way as the training images were
    if checkpoint.get("fix_data", False) != FIX_DATA:
        raise ValueError(f"Checkpoint was trained with fix_data={checkpoint.get('fix_data', False)}, "
                         f"but fix_data is {FIX_DATA} now. Set NERF_FIX_DATA to match.")

    sampler = checkpoint.get("sampler") or {}
    if "n_first" in sampler:
        settings = SamplerSettings(**sampler)
        nerf_kwargs = checkpoint["nerf_kwargs"]
    else:
        # Checkpoints from before the proposal/spacing changes: book model, linear spacing
        is_hier = checkpoint.get("fine_state_dict") is not None
        settings = SamplerSettings(
            type="hierarchical" if is_hier else "uniform",
            n_first=sampler.get("n_pts_first_pass", 128),
            n_fine=sampler.get("n_fine", 0),
            min_depth=RAY_MIN_DEPTH,
            max_depth=RAY_MAX_DEPTH,
        )
        nerf_kwargs = {"density_uses_delta": sampler.get("density_uses_delta", DENSITY_USES_DELTA),
                       "architecture": "book", "n_harmonic_functions": 60, "n_harmonic_functions_dir": 60,
                       "omega0": 0.1, "include_input": False, "appearance_dim": 0, "contract": False}

    appearance_state = checkpoint.get("appearance_state_dict")
    n_codes = appearance_state["weight"].shape[0] if appearance_state is not None else 0
    first, second, appearance = build_models(
        settings, nerf_kwargs, checkpoint.get("proposal_kwargs") or {}, n_codes, device
    )

    # Older baseline-style checkpoints only have model_state_dict, which is the coarse model
    if settings.type == "proposal":
        first.load_state_dict(checkpoint["proposal_state_dict"])
    else:
        first.load_state_dict(checkpoint.get("coarse_state_dict") or checkpoint["model_state_dict"])
    if second is not None:
        second.load_state_dict(checkpoint["fine_state_dict"])

    appearance_weights = None
    if appearance is not None:
        appearance.load_state_dict(appearance_state)
        appearance_weights = appearance.weight

    for model in (first, second):
        if model is not None:
            model.eval()
    print(f"Loaded checkpoint from iteration {checkpoint['iteration']} ({settings.type} sampling, "
          f"spacing={settings.spacing}, {nerf_kwargs.get('architecture', 'book')} model)")
    return first, second, appearance_weights, settings


def test_view_codes(appearance_weights, view_indices):
    """Appearance code for every test view. Held-out views were never seen in training, so they
    have no code of their own (NeRF-W, Martin-Brualla et al. 2021):

    mean:      the average of all training codes (the same for every test view)
    neighbour: the average code of the training frame(s) recorded closest in time. Poster is a
               video, so these were taken a fraction of a second before / after the test frame,
               with the same exposure and white balance. Uses training information only.
    """
    if EVAL_APPEARANCE == "mean":
        return appearance_weights.mean(dim=0, keepdim=True).expand(len(view_indices), -1)
    if EVAL_APPEARANCE != "neighbour":
        raise ValueError(f"Unknown evaluation.appearance_code '{EVAL_APPEARANCE}', expected mean or neighbour")
    if not TEST_EVERY:
        raise ValueError("appearance_code: neighbour needs a dataset split with test_every (a video)")

    # The frames in transforms.json are not stored in recording order, so the time order comes from
    # the frame number in the file name (frame_00216.png -> 216)
    with open(os.path.join(DATA_DIR, "transforms.json"), encoding="utf-8") as f:
        frames = json.load(f)["frames"]
    numbers = [int(re.findall(r"\d+", os.path.basename(fr["file_path"]))[-1]) for fr in frames]
    # Same split as train.py: appearance code r belongs to the r-th frame that is not held out
    train_positions = [i for i in range(len(frames)) if i % TEST_EVERY != 0]
    train_numbers = torch.tensor([numbers[i] for i in train_positions])

    codes = []
    for v in view_indices:
        distance = (train_numbers - numbers[v]).abs()
        closest = (distance == distance.min()).nonzero().flatten().to(appearance_weights.device)
        codes.append(appearance_weights[closest].mean(dim=0))
    return torch.stack(codes)


def write_per_view_csv(rows, path):
    """Saves PSNR, SSIM and LPIPS for every test view."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["view_index", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(rows, path):
    """Saves the mean and standard deviation over all test views in one row."""
    summary = {"dataset": ACTIVE_DATASET, "n_views": len(rows)}
    for key in ("psnr", "ssim", "lpips"):
        values = [r[key] for r in rows]
        summary[f"{key}_mean"] = statistics.mean(values)
        # The standard deviation needs at least two views
        summary[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def pick_figure_views(rows, n):
    """Picks n views spread over the PSNR ranking, always including the best and the worst."""
    # Positions in rows sorted from highest to lowest PSNR
    order = sorted(range(len(rows)), key=lambda i: rows[i]["psnr"], reverse=True)
    n = min(n, len(order))
    if n == 1:
        return [order[0]]
    # Evenly spaced ranks from the best to the worst view
    ranks = [round(k * (len(order) - 1) / (n - 1)) for k in range(n)]
    return [order[r] for r in ranks]


def save_qualitative_figure(targets, renders, rows, positions, path):
    """Saves a figure with ground truth, render and absolute error for the selected views."""
    height, width = targets[0].shape[:2]
    n_cols = len(positions)
    extra_w, extra_h = 0.9, 0.5  # room for the row labels, colour bar and column labels

    # Fits the figure to the page width, but shrinks it if it gets too tall
    col_width = (IEEE_TEXT_WIDTH_IN - extra_w) / n_cols
    if 3 * col_width * height / width + extra_h > MAX_FIGURE_HEIGHT_IN:
        col_width = (MAX_FIGURE_HEIGHT_IN - extra_h) / 3 * width / height
    fig_size = (n_cols * col_width + extra_w, 3 * col_width * height / width + extra_h)

    # Error per pixel averaged over the RGB channels
    errors = [(renders[p] - targets[p]).abs().mean(dim=-1) for p in positions]
    vmax = max(float(e.max()) for e in errors)  # one shared scale, so the columns are comparable

    # Extra narrow column for the colour bar, so every image column has the same width
    fig, axes = plt.subplots(
        3, n_cols + 1, figsize=fig_size, squeeze=False,
        gridspec_kw={"wspace": 0.04, "hspace": 0.04, "width_ratios": [1] * n_cols + [0.06]},
    )
    for col, p in enumerate(positions):
        axes[0, col].imshow(targets[p].numpy())
        axes[1, col].imshow(renders[p].numpy())
        im = axes[2, col].imshow(errors[col].numpy(), cmap="Reds", vmin=0.0, vmax=vmax)
        axes[2, col].set_xlabel(
            f"view {rows[p]['view_index']}\n{rows[p]['psnr']:.2f} dB", fontsize=7
        )
    for row, label in enumerate(("Ground truth", "Render", "|Error|")):
        axes[row, 0].set_ylabel(label, fontsize=8)
    # Hides ticks and borders around the images
    for ax in axes[:, :n_cols].ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Only the bottom cell of the last column holds the colour bar
    axes[0, n_cols].axis("off")
    axes[1, n_cols].axis("off")
    cbar = fig.colorbar(im, cax=axes[2, n_cols])
    cbar.ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main():
    """Renders every test view, computes the metrics and saves the results and the figure."""
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Results with the default (mean) appearance code keep the normal file names, other modes
    # get a suffix, so both can be reported side by side
    suffix = "" if EVAL_APPEARANCE == "mean" else f"_{EVAL_APPEARANCE}"
    render_dir = OUTPUT_DIR / f"test_renders{suffix}"
    render_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {ACTIVE_DATASET}")
    target_images, target_cameras, view_indices = load_test_split(device)
    n_views, height, width, _ = target_images.shape
    print(f"Evaluating on {n_views} test views at {width}x{height}")

    first, second, appearance_weights, settings = load_model(device)
    lpips_metric = LPIPSMetric(device)
    codes = None
    if appearance_weights is not None:
        codes = test_view_codes(appearance_weights, view_indices)
        print(f"Appearance code for the test views: {EVAL_APPEARANCE}")

    # One ray per pixel. The first-pass sample count and depth range must match training, and there
    # is no random jitter at test time (training=False), so results are deterministic
    raysampler = NDCMultinomialRaysampler(
        image_height=height,
        image_width=width,
        n_pts_per_ray=settings.n_first,
        min_depth=settings.min_depth,
        max_depth=settings.max_depth,
    )
    raymarcher = EmissionAbsorptionRaymarcher()

    rows, renders, targets = [], [], []
    for i in tqdm(range(n_views), desc="Rendering test views"):
        ray_bundle = raysampler(cameras=select_cameras(target_cameras, [i]))
        # First pass, then (if hierarchical / proposal) the second pass; chunked to save memory
        out = render_rays(
            ray_bundle, first, second, raymarcher, settings, training=False, chunked=True,
            appearance_codes=codes[i:i + 1] if codes is not None else None,
        )
        # Keeps the RGB channels and drops the opacity channel
        prediction = out["rgba"][0, ..., :3].clamp(0.0, 1.0)
        target = target_images[i]

        rows.append({
            "view_index": view_indices[i],
            "psnr": psnr(prediction, target),
            "ssim": ssim(prediction, target),
            "lpips": lpips_metric(prediction, target),
        })
        renders.append(prediction.cpu())
        targets.append(target.cpu())
        plt.imsave(render_dir / f"view_{view_indices[i]:03d}.png", prediction.cpu().numpy())

    write_per_view_csv(rows, OUTPUT_DIR / f"metrics_per_view{suffix}.csv")
    summary = write_summary_csv(rows, OUTPUT_DIR / f"metrics{suffix}.csv")

    figure_path = FIGURES_DIR / f"qualitative_{ACTIVE_DATASET}_{EXPERIMENT}{suffix}.png"
    save_qualitative_figure(targets, renders, rows, pick_figure_views(rows, N_FIGURE_VIEWS), figure_path)

    print(
        f"PSNR  {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f} dB\n"
        f"SSIM  {summary['ssim_mean']:.3f} ± {summary['ssim_std']:.3f}\n"
        f"LPIPS {summary['lpips_mean']:.3f} ± {summary['lpips_std']:.3f}\n"
        f"Saved metrics to {OUTPUT_DIR} and figure to {figure_path}"
    )


if __name__ == "__main__":
    main()