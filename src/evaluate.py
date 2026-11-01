"""
Evaluates a trained NeRF on the held-out test views of the active dataset.
Run from the project root after training with python -m src.evaluate
Works for both uniform (coarse only) and hierarchical (coarse + fine) checkpoints.
"""
import csv
import statistics

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pytorch3d.renderer import (
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
)

from src.load_nerf_data import load_nerf_data
from src.model import NeuralRadianceField
from src.metrics import psnr, ssim, LPIPSMetric
from src.render import render_rays
from src.train import select_cameras, fix_fov_for_aspect
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
    N_PTS_PER_RAY,
    N_COARSE,
    N_FINE,
    TEST_EVERY,
    MAX_TEST_VIEWS,
    N_FIGURE_VIEWS,
    EXPERIMENT,
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
        )
        all_idx = torch.arange(len(images))
        test_idx = all_idx[all_idx % TEST_EVERY == 0]
    # Lego comes with its own test split
    else:
        images, _, cameras = load_nerf_data(
            str(DATA_DIR), split="test", device=device,
            znear=CAMERA_ZNEAR, zfar=CAMERA_ZFAR, resize_to=RESIZE_TO,
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
    """Loads the coarse model, and the fine model if the run used hierarchical sampling."""
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

    # Older baseline-style checkpoints only have model_state_dict, which is the coarse model
    coarse_state = checkpoint.get("coarse_state_dict", checkpoint.get("model_state_dict"))
    coarse = NeuralRadianceField().to(device)
    coarse.load_state_dict(coarse_state)
    coarse.eval()

    # Only hierarchical runs have a fine model
    fine = None
    if checkpoint.get("fine_state_dict") is not None:
        fine = NeuralRadianceField().to(device)
        fine.load_state_dict(checkpoint["fine_state_dict"])
        fine.eval()

    sampling = "hierarchical" if fine is not None else "uniform"
    print(f"Loaded checkpoint from iteration {checkpoint['iteration']} ({sampling} sampling)")
    return coarse, fine


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
    render_dir = OUTPUT_DIR / "test_renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {ACTIVE_DATASET}")
    target_images, target_cameras, view_indices = load_test_split(device)
    n_views, height, width, _ = target_images.shape
    print(f"Evaluating on {n_views} test views at {width}x{height}")

    coarse, fine = load_model(device)
    lpips_metric = LPIPSMetric(device)

    # One ray per pixel. The coarse sample count must match training, and there is no
    # random jitter at test time (stratified sampling is off by default), so results are deterministic
    raysampler = NDCMultinomialRaysampler(
        image_height=height,
        image_width=width,
        n_pts_per_ray=N_COARSE if fine is not None else N_PTS_PER_RAY,
        min_depth=RAY_MIN_DEPTH,
        max_depth=RAY_MAX_DEPTH,
    )
    raymarcher = EmissionAbsorptionRaymarcher()

    rows, renders, targets = [], [], []
    for i in tqdm(range(n_views), desc="Rendering test views"):
        ray_bundle = raysampler(cameras=select_cameras(target_cameras, [i]))
        # Coarse pass, then (if hierarchical) the fine pass; chunked to save memory
        rendered, _ = render_rays(
            ray_bundle, coarse, fine, raymarcher, N_FINE, training=False, chunked=True
        )
        # Keeps the RGB channels and drops the opacity channel
        prediction = rendered[0, ..., :3].clamp(0.0, 1.0)
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

    write_per_view_csv(rows, OUTPUT_DIR / "metrics_per_view.csv")
    summary = write_summary_csv(rows, OUTPUT_DIR / "metrics.csv")

    figure_path = FIGURES_DIR / f"qualitative_{ACTIVE_DATASET}_{EXPERIMENT}.png"
    save_qualitative_figure(targets, renders, rows, pick_figure_views(rows, N_FIGURE_VIEWS), figure_path)

    print(
        f"PSNR  {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f} dB\n"
        f"SSIM  {summary['ssim_mean']:.3f} ± {summary['ssim_std']:.3f}\n"
        f"LPIPS {summary['lpips_mean']:.3f} ± {summary['lpips_std']:.3f}\n"
        f"Saved metrics to {OUTPUT_DIR} and figure to {figure_path}"
    )


if __name__ == "__main__":
    main()