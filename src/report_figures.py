"""
Builds the ablation figures and tables for the report from every finished run.
Run from the project root after train.py and evaluate.py with python -m src.report_figures

It collects:
- the frozen baseline:   outputs/baseline/<dataset>/
- every src experiment:  outputs/src/<dataset>/<experiment>/
and writes, per dataset, to the figures folder:
- table_ablation_<dataset>.tex   LaTeX table: sampler settings, PSNR, SSIM, LPIPS, training time
- loss_<dataset>.pdf             colour loss against iteration and against wall-clock time
"""
import csv

import matplotlib.pyplot as plt
import yaml
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from src.repo_util.LoadConfigurations import (
    PROJECT_ROOT,
    DATASET_NAMES,
    DATASET_OUTPUT_DIRS,
    FIGURES_DIR,
    LOSS_SMOOTHING_WINDOW,
    FIX_DATA,
)

IEEE_COLUMN_WIDTH_IN = 3.5   # width of one column in an IEEE two-column paper in inches
IEEE_TEXT_WIDTH_IN = 7.16    # full page width
TEXT_COLOR = "#333333"
GRID_COLOR = "#e0e0e0"
BASELINE_NAME = "A2 baseline"

# Distinct colours for up to 8 runs, the baseline is always grey
RUN_COLORS = ["#2a78d6", "#eb6834", "#2ca02c", "#9467bd", "#d62728", "#17becf", "#bcbd22", "#8c564b"]
BASELINE_COLOR = "#888888"


# Reading results

def find_runs(dataset):
    """Returns a list of runs for one dataset: the baseline first, then the src experiments by name."""
    runs = []
    baseline_dir = PROJECT_ROOT / "outputs" / "baseline" / dataset
    # The A2 baseline was trained and evaluated without the data fixes, so its numbers are not
    # comparable with runs on the corrected data. There the src 'uniform' run (book model) is the reference
    has_baseline = (baseline_dir / "metrics.csv").exists() or (baseline_dir / "loss_history.csv").exists()
    if has_baseline and not FIX_DATA:
        runs.append({"name": BASELINE_NAME, "dir": baseline_dir, "is_baseline": True})

    src_dir = DATASET_OUTPUT_DIRS[dataset]  # outputs/src/<dataset>, one subfolder per experiment
    if src_dir.exists():
        for exp_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
            if (exp_dir / "metrics.csv").exists() or (exp_dir / "loss_history.csv").exists():
                runs.append({"name": exp_dir.name, "dir": exp_dir, "is_baseline": False})
    return runs


def read_csv_rows(path):
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_metrics(run):
    """Mean and standard deviation from metrics.csv, or None if the run was not evaluated."""
    rows = read_csv_rows(run["dir"] / "metrics.csv")
    return rows[0] if rows else None


def read_loss_history(run):
    """Colour loss per iteration, and elapsed seconds if the run logged them (src runs do)."""
    rows = read_csv_rows(run["dir"] / "loss_history.csv")
    if not rows:
        return None, None
    losses = [float(r["color_loss"]) for r in rows]
    if "elapsed_s" in rows[0] and rows[0]["elapsed_s"]:
        seconds = [float(r["elapsed_s"]) for r in rows]
    else:
        seconds = None
    return losses, seconds


def read_sampler_settings(run):
    """Sampler settings from the config saved with the run. The baseline is plain uniform sampling."""
    cfg_path = run["dir"] / "config_used.yaml"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    n_pts = cfg.get("render", {}).get("n_pts_per_ray", "?")
    sampler = cfg.get("sampler")
    if run["is_baseline"] or not sampler:
        return {"type": "uniform", "stratified": False, "delta": False, "points": str(n_pts), "extras": "--"}
    if sampler.get("type") == "hierarchical":
        points = f"{sampler.get('n_coarse')}+{sampler.get('n_fine')}"
    elif sampler.get("type") == "proposal":
        points = f"{sampler.get('n_coarse')}$\\to${sampler.get('n_fine')}"
    else:
        points = str(n_pts)
    return {
        "type": sampler.get("type", "uniform"),
        "stratified": bool(sampler.get("stratified", False)),
        "delta": bool(sampler.get("density_uses_delta", False)),
        "points": points,
        "extras": read_extras(cfg),
    }


def read_extras(cfg):
    """Short description of the final project additions a run used, '--' if none (book model)."""
    model, sampler, train = cfg.get("model", {}), cfg.get("sampler", {}), cfg.get("train", {})
    render = cfg.get("render", {})
    extras = []
    if model.get("architecture", "book") == "nerf":
        extras.append("8-layer MLP")
    n_pos = model.get("n_harmonic_functions", 60)
    n_dir = model.get("n_harmonic_functions_dir", n_pos)
    if model.get("architecture", "book") == "hashgrid":
        extras.append("hash grid")  # replaces the positional encoding of the positions
    elif (n_pos, n_dir) != (60, 60):
        extras.append(f"L={n_pos}/{n_dir}")
    if sampler.get("spacing", "linear") != "linear":
        extras.append(sampler["spacing"])
    if sampler.get("contract", False):
        extras.append("contract")
    if float(train.get("distortion_weight", 0.0)) > 0:
        extras.append("dist.")
    if int(model.get("appearance_dim", 0)) > 0:
        extras.append("app.")
    if model.get("time_input", False):
        extras.append("time input")
    if train.get("lr_schedule", "step") == "exp":
        extras.append("exp. LR")
    n_rays = int(train.get("batch_size", 4)) * int(render.get("n_rays_per_image", 400))
    if n_rays != 1600:
        extras.append(f"{n_rays} rays")
    return ", ".join(extras) if extras else "--"


# Table

def latex_escape(text):
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def format_minutes(seconds):
    return "--" if seconds is None else f"{seconds / 60:.1f}"


def write_ablation_table(dataset, runs):
    """One LaTeX table per dataset, one row per run, best value per metric in bold."""
    rows = []
    for run in runs:
        metrics = read_metrics(run)
        if metrics is None:
            continue
        _, seconds = read_loss_history(run)
        rows.append({
            "run": run,
            "metrics": metrics,
            "sampler": read_sampler_settings(run),
            "train_s": seconds[-1] if seconds else None,
        })
    if not rows:
        print(f"[{dataset}] No metrics.csv found - run evaluate first. Skipping the table.")
        return

    # Best value per metric, so it can be printed in bold
    best = {
        "psnr": max(float(r["metrics"]["psnr_mean"]) for r in rows),
        "ssim": max(float(r["metrics"]["ssim_mean"]) for r in rows),
        "lpips": min(float(r["metrics"]["lpips_mean"]) for r in rows),
    }

    def cell(m, key, digits):
        mean, std = float(m[key + "_mean"]), float(m[key + "_std"])
        value = f"{mean:.{digits}f} \\pm {std:.{digits}f}"
        return f"$\\mathbf{{{value}}}$" if abs(mean - best[key]) < 1e-12 else f"${value}$"

    tick = lambda b: r"\checkmark" if b else "--"
    body = "\n".join(
        f"{latex_escape(r['run']['name'])} & {r['sampler']['type']} & {r['sampler']['points']} & "
        f"{tick(r['sampler']['delta'])} & {tick(r['sampler']['stratified'])} & "
        f"{r['sampler']['extras']} & "
        f"{cell(r['metrics'], 'psnr', 2)} & {cell(r['metrics'], 'ssim', 3)} & "
        f"{cell(r['metrics'], 'lpips', 3)} & {format_minutes(r['train_s'])} \\\\"
        for r in rows
    )
    n_views = rows[0]["metrics"]["n_views"]
    data_note = (" Corrected data: lego colours multiplied by alpha, poster images undistorted."
                 if FIX_DATA else "")
    table = rf"""% Generated by src/report_figures.py - do not edit by hand, rerun the script.
% Needs \usepackage{{booktabs}} and \usepackage{{amssymb}}. Include with \input{{figures/table_ablation_{dataset}.tex}}
\begin{{table*}}[t]
\centering
\caption{{Ablation on {dataset.capitalize()} ({n_views} held-out test views, mean $\pm$ standard deviation).{data_note}
Points: samples per ray (hierarchical: coarse + fine, proposal: proposal $\to$ NeRF).
$\delta$: opacity uses the sample spacing (NeRF eq.~3). Strat.: stratified sampling during training.
Other: 8-layer NeRF MLP or hash grid (Instant-NGP), positional encoding frequencies (position/direction), sample spacing,
scene contraction, distortion loss, appearance embeddings, frame time as input (T-NeRF), exponential learning-rate decay and
rays per training step if not 1600 (-- = book model).
Time: wall-clock training time in minutes (-- if not logged).}}
\label{{tab:ablation_{dataset}}}
\begin{{tabular}}{{lllcclcccc}}
\toprule
Run & Sampler & Points & $\delta$ & Strat. & Other & PSNR $\uparrow$ (dB) & SSIM $\uparrow$ & LPIPS $\downarrow$ & Time (min) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    path = FIGURES_DIR / f"table_ablation_{dataset}.tex"
    path.write_text(table, encoding="utf-8")
    print(f"[{dataset}] Saved {path}")

    # Plain-text version in the terminal, handy for the experiment log
    print(f"[{dataset}] {'run':<24} {'PSNR':>7} {'SSIM':>6} {'LPIPS':>6} {'min':>6}")
    for r in rows:
        m = r["metrics"]
        print(f"[{dataset}] {r['run']['name']:<24} {float(m['psnr_mean']):7.2f} "
              f"{float(m['ssim_mean']):6.3f} {float(m['lpips_mean']):6.3f} {format_minutes(r['train_s']):>6}")


# Loss curves

def moving_average(values, window):
    """Smooths the loss with a moving average over the last window iterations."""
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v
        # Removes the value that falls out of the window
        if i >= window:
            total -= values[i - window]
        # The first points average over the values available so far
        out.append(total / min(i + 1, window))
    return out


def style_axis(ax):
    ax.set_yscale("log")
    # Log scale with plain number labels at 0.02, 0.05, 0.1, 0.2 and so on
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    # Light grid and only the left and bottom borders
    ax.grid(True, which="both", axis="y", color=GRID_COLOR, linewidth=0.5)
    ax.grid(True, which="major", axis="x", color=GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
    ax.tick_params(which="both", colors=TEXT_COLOR, labelsize=7)


def plot_loss_curves(dataset, runs):
    """Colour loss per run, against iteration (left) and against wall-clock time (right)."""
    curves = []
    color_idx = 0
    for run in runs:
        losses, seconds = read_loss_history(run)
        if not losses:
            continue
        if run["is_baseline"]:
            color = BASELINE_COLOR
        else:
            color = RUN_COLORS[color_idx % len(RUN_COLORS)]
            color_idx += 1
        curves.append((run, moving_average(losses, LOSS_SMOOTHING_WINDOW), seconds, color))
    if not curves:
        print(f"[{dataset}] No loss_history.csv found - train first. Skipping loss curves.")
        return

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})
    fig, (ax_it, ax_time) = plt.subplots(1, 2, figsize=(IEEE_TEXT_WIDTH_IN, 2.4))

    for run, smoothed, seconds, color in curves:
        style = "--" if run["is_baseline"] else "-"
        ax_it.plot(range(len(smoothed)), smoothed, style, color=color, linewidth=1.1, label=run["name"])
        # Only runs that logged their time can go on the time axis (the baseline did not)
        if seconds:
            ax_time.plot([s / 60 for s in seconds], smoothed, style, color=color, linewidth=1.1,
                         label=run["name"])

    ax_it.set_xlabel("Iteration")
    ax_it.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x else "0"))
    ax_time.set_xlabel("Training time (min)")
    for ax in (ax_it, ax_time):
        style_axis(ax)
    ax_it.set_ylabel(f"Colour loss ({LOSS_SMOOTHING_WINDOW}-it. avg.)")
    ax_it.set_title(f"{dataset.capitalize()}: loss per iteration", fontsize=8, loc="left", color=TEXT_COLOR)
    ax_time.set_title("Loss per wall-clock time", fontsize=8, loc="left", color=TEXT_COLOR)
    ax_it.legend(fontsize=6, frameon=False)

    fig.tight_layout()
    path = FIGURES_DIR / f"loss_{dataset}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[{dataset}] Saved {path}")


def main():
    """Builds the ablation table and loss curves for every dataset that has results."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in DATASET_NAMES:
        runs = find_runs(dataset)
        if not runs:
            print(f"[{dataset}] No runs found.")
            continue
        print(f"[{dataset}] Runs: {', '.join(r['name'] for r in runs)}")
        write_ablation_table(dataset, runs)
        plot_loss_curves(dataset, runs)


if __name__ == "__main__":
    main()