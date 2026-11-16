"""
Renders a 360-degree orbit around the scene with trained models, for the report and the presentation.
Run from the project root after training, for example:

    NERF_DATASET=lego python -m src.render_video                         # worst and best model
    NERF_DATASET=poster python -m src.render_video --runs uniform hashgrid

The orbit goes around the point the training cameras look at, at their average distance and
height, so it stays inside the region the models were trained on. Rendering uses the same
PyTorch3D raysampler and raymarcher as training and evaluation (render_rays).
- Time models (dynamic lego): the time runs from 0 to 1 during the orbit, so the bucket moves.
- Appearance models (poster): every frame blends the codes of the 4 nearest training cameras.

Writes to <figures dir>/orbits/:
    <dataset>_<run>_orbit.gif          one model
    <dataset>_orbit_comparison.gif     all models side by side (presentation)
    <dataset>_orbit_filmstrip.png      a few orbit frames per model, plus the depth of the last
                                       model (for the report, where GIFs do not animate)
"""
import argparse

import matplotlib
matplotlib.use("Agg")  # saves figures to disk without opening windows
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
    look_at_view_transform,
)

from src.evaluate import load_model
from src.render import render_rays
from src.train import load_train_split
from src.repo_util.LoadConfigurations import (
    ACTIVE_DATASET,
    DATASET_OUTPUT_DIRS,
    FIGURES_DIR,
    CAMERA_ZNEAR,
    CAMERA_ZFAR,
)

# Worst (book model) and best model of the final ablation, per dataset
DEFAULT_RUNS = {"lego": ["uniform", "hashgrid_time"], "poster": ["uniform", "hashgrid"]}
LABELS = {
    "uniform": "Book model\n(baseline)",
    "hashgrid": "Hash grid",
    "hashgrid_time": "Hash grid\n+ time",
}
IEEE_TEXT_WIDTH_IN = 7.16
MIN_GIF_HEIGHT = 400  # small renders (lego is 200 px) are scaled up in the GIFs


def orbit_poses(cameras, n_frames):
    """Camera rotations and translations on a circle around the scene.

    In PyTorch3D the columns of R are the camera axes in world coordinates: column 1 is the
    camera's up direction and column 2 its viewing direction.
    """
    centers = cameras.get_camera_center()                                   # (N, 3)
    up = torch.nn.functional.normalize(cameras.R[:, :, 1].mean(dim=0), dim=0)
    forward = torch.nn.functional.normalize(cameras.R[:, :, 2], dim=-1)

    # The point all viewing rays pass closest to (least squares), i.e. what the cameras look at
    eye3 = torch.eye(3, device=centers.device)
    projections = eye3 - forward[:, :, None] * forward[:, None, :]          # (N, 3, 3)
    target = torch.linalg.solve(projections.sum(0), (projections @ centers[..., None]).sum(0))[:, 0]

    # Average height above the target and distance from the up axis
    offsets = centers - target
    heights = offsets @ up
    flat = offsets - heights[:, None] * up
    radius = flat.norm(dim=-1).mean()
    e1 = torch.nn.functional.normalize(flat[0], dim=0)  # starts at the first training camera
    e2 = torch.linalg.cross(up, e1)

    angles = torch.linspace(0, 2 * np.pi, n_frames + 1, device=centers.device)[:-1]
    eyes = target + radius * (torch.cos(angles)[:, None] * e1 + torch.sin(angles)[:, None] * e2) \
        + heights.mean() * up
    R, T = look_at_view_transform(eye=eyes, at=target.expand_as(eyes), up=up.expand_as(eyes),
                                  device=centers.device)
    return R, T, eyes, centers


def frame_codes(appearance_weights, eyes, train_centers, k=4):
    """Appearance code per orbit frame: inverse-distance blend of the k nearest training cameras,
    so the colours change smoothly along the orbit."""
    distances = torch.cdist(eyes, train_centers)                            # (F, N)
    nearest, idx = distances.topk(k, dim=-1, largest=False)
    weights = 1.0 / nearest.clamp_min(1e-6)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return (weights[..., None] * appearance_weights[idx]).sum(dim=1)        # (F, dim)


@torch.no_grad()
def render_orbit(run, R, T, fov, height, width, eyes, train_centers, device):
    """Renders every orbit frame with one trained run. Returns RGB, depth and opacity per frame."""
    checkpoint = DATASET_OUTPUT_DIRS[ACTIVE_DATASET] / run / "checkpoint.pt"
    first, second, appearance_weights, settings = load_model(device, checkpoint)
    nerf = second if second is not None else first
    n_frames = R.shape[0]

    codes = frame_codes(appearance_weights, eyes, train_centers) if appearance_weights is not None else None
    times = torch.linspace(0, 1, n_frames, device=device) if getattr(nerf, "time_input", False) else None

    raysampler = NDCMultinomialRaysampler(
        image_height=height, image_width=width, n_pts_per_ray=settings.n_first,
        min_depth=settings.min_depth, max_depth=settings.max_depth,
    )
    raymarcher = EmissionAbsorptionRaymarcher()
    rgbs, depths, opacities = [], [], []
    for i in tqdm(range(n_frames), desc=f"Orbit {run}"):
        camera = FoVPerspectiveCameras(R=R[i:i + 1], T=T[i:i + 1], fov=fov, znear=CAMERA_ZNEAR,
                                       zfar=CAMERA_ZFAR, device=device)
        out = render_rays(
            raysampler(cameras=camera), first, second, raymarcher, settings, training=False, chunked=True,
            appearance_codes=codes[i:i + 1] if codes is not None else None,
            times=times[i:i + 1] if times is not None else None,
        )
        opacity = out["rgba"][0, ..., 3]
        # Expected depth of the ray, divided by the opacity so it is not pulled towards 0 at edges
        depth = (out["weights"][0] * out["lengths"][0]).sum(dim=-1) / opacity.clamp_min(1e-6)
        rgbs.append(out["rgba"][0, ..., :3].clamp(0, 1).cpu().numpy())
        depths.append(depth.cpu().numpy())
        opacities.append(opacity.cpu().numpy())
    return np.stack(rgbs), np.stack(depths), np.stack(opacities)


def depth_to_color(depths, opacities):
    """Disparity (1 / depth) in the turbo colour map, near = red, far = blue, empty space black.
    One scale for the whole orbit, so the frames are comparable."""
    disparity = 1.0 / np.maximum(depths, 1e-3)
    valid = opacities > 0.5
    lo, hi = np.percentile(disparity[valid], [2, 98]) if valid.any() else (0.0, 1.0)
    colored = plt.get_cmap("turbo")(np.clip((disparity - lo) / max(hi - lo, 1e-6), 0, 1))[..., :3]
    return colored * valid[..., None]


def to_pil(image, scale):
    pil = Image.fromarray((image * 255).round().astype(np.uint8))
    return pil.resize((pil.width * scale, pil.height * scale), Image.BICUBIC) if scale > 1 else pil


def save_gif(frames, path, fps):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0)
    print(f"Saved {path}")


def save_comparison_gif(videos, labels, path, fps, scale):
    """All models side by side with a label above each, one GIF frame per orbit frame."""
    gap, bar = 8, 28
    frames = []
    for i in range(len(videos[0])):
        tiles = [to_pil(video[i], scale) for video in videos]
        w, h = tiles[0].size
        canvas = Image.new("RGB", (len(tiles) * w + (len(tiles) - 1) * gap, h + bar), "white")
        draw = ImageDraw.Draw(canvas)
        for j, (tile, label) in enumerate(zip(tiles, labels)):
            x = j * (w + gap)
            canvas.paste(tile, (x, bar))
            draw.text((x + 6, 7), label, fill="black")
        frames.append(canvas)
    save_gif(frames, path, fps)


def save_filmstrip(rows, row_labels, path, n_cols=6):
    """Report figure: n_cols evenly spaced orbit frames (columns) for every row."""
    n_frames = len(rows[0])
    cols = np.linspace(0, n_frames, n_cols, endpoint=False).round().astype(int)
    height, width = rows[0][0].shape[:2]
    col_w = (IEEE_TEXT_WIDTH_IN - 0.4) / n_cols
    fig, axes = plt.subplots(len(rows), n_cols, squeeze=False,
                             figsize=(IEEE_TEXT_WIDTH_IN, len(rows) * col_w * height / width + 0.3),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.03})
    for r, (row, label) in enumerate(zip(rows, row_labels)):
        for c, frame in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(row[frame])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == len(rows) - 1:
                ax.set_xlabel(f"{360 * frame / n_frames:.0f}°", fontsize=7)
        axes[r, 0].set_ylabel(label, fontsize=7)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS.get(ACTIVE_DATASET),
                        help="experiments to render, worst first (the depth row uses the last one)")
    parser.add_argument("--frames", type=int, default=72, help="frames in the orbit (72 = every 5 degrees)")
    parser.add_argument("--fps", type=int, default=18)
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    out_dir = FIGURES_DIR / "orbits"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training cameras (same image size and field of view correction as in training)
    images, _, train_cameras = load_train_split(device)
    height, width = images.shape[1:3]
    del images
    fov = train_cameras.fov[:1]
    R, T, eyes, train_centers = orbit_poses(train_cameras, args.frames)
    scale = max(1, round(MIN_GIF_HEIGHT / height))

    videos, depth_video = [], None
    for run in args.runs:
        rgbs, depths, opacities = render_orbit(run, R, T, fov, height, width, eyes, train_centers, device)
        videos.append(rgbs)
        save_gif([to_pil(f, scale) for f in rgbs], out_dir / f"{ACTIVE_DATASET}_{run}_orbit.gif", args.fps)
        depth_video = depth_to_color(depths, opacities)  # kept for the last (best) run

    labels = [LABELS.get(run, run) for run in args.runs]
    save_comparison_gif(videos, [label.replace("\n", " ") for label in labels],
                        out_dir / f"{ACTIVE_DATASET}_orbit_comparison.gif", args.fps, scale)
    save_filmstrip(videos + [depth_video], labels + ["Depth\n(" + labels[-1].split("\n")[0] + ")"],
                   out_dir / f"{ACTIVE_DATASET}_orbit_filmstrip.png")


if __name__ == "__main__":
    main()
