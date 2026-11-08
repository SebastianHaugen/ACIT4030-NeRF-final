# Imports
import os
from pathlib import Path
import yaml


# Configuration file
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # two folders up from src/repo_util
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


# Retrieving configurations as constants

# Preview
PREVIEW_MESSAGE = CONFIG["preview"]["the-preview"]

# Active dataset (lego | poster), the NERF_DATASET environment variable overrides
# the config value, so a run can pick its dataset without editing config.yaml
ACTIVE_DATASET = os.environ.get("NERF_DATASET", CONFIG["active_dataset"])
if ACTIVE_DATASET not in CONFIG["datasets"]:
    raise ValueError(f"Unknown dataset '{ACTIVE_DATASET}', expected one of {list(CONFIG['datasets'])}")
_DATASET = CONFIG["datasets"][ACTIVE_DATASET]  # settings of the active dataset only

# Model (src/model.py)
N_HARMONIC_FUNCTIONS = CONFIG["model"]["n_harmonic_functions"]
HARMONIC_OMEGA0 = CONFIG["model"]["omega0"]
N_HIDDEN_NEURONS = CONFIG["model"]["n_hidden_neurons"]

# Rendering
N_RAYS_PER_IMAGE = CONFIG["render"]["n_rays_per_image"]
N_PTS_PER_RAY = CONFIG["render"]["n_pts_per_ray"]  # may be overridden below by NERF_N_PTS
N_BATCHES_FULL_RENDER = CONFIG["render"]["n_batches_full_render"]

# Training
BATCH_SIZE = CONFIG["train"]["batch_size"]
LEARNING_RATE = CONFIG["train"]["learning_rate"]
LR_DECAY_AT = CONFIG["train"]["lr_decay_at"]
LR_DECAY_FACTOR = CONFIG["train"]["lr_decay_factor"]
HUBER_SCALING = CONFIG["train"]["huber_scaling"]
RANDOM_SEED = CONFIG["train"]["seed"]
VISUALIZE_EVERY = CONFIG["train"]["vis_every"]
CHECKPOINT_EVERY = CONFIG["train"]["checkpoint_every"]

# Dataset-specific (values of the active dataset)
DATA_DIR = PROJECT_ROOT / _DATASET["data_dir"]
OUTPUT_DIR = PROJECT_ROOT / _DATASET["output_dir"]
N_ITERATIONS = _DATASET["n_iter"]
RESIZE_TO = tuple(_DATASET["resize_to"]) if _DATASET["resize_to"] else None  # None keeps full resolution
CAMERA_ZNEAR = _DATASET["znear"]
CAMERA_ZFAR = _DATASET["zfar"]
RAY_MIN_DEPTH = _DATASET["min_depth"]
RAY_MAX_DEPTH = _DATASET["max_depth"]
USE_SILHOUETTE_LOSS = _DATASET["use_silhouette_loss"]
TEST_EVERY = _DATASET["test_every"]
N_FIGURE_VIEWS = _DATASET["n_figure_views"]

# Evaluation (src/evaluate.py, src/report_figures.py)
MAX_TEST_VIEWS = CONFIG["evaluation"]["max_test_views"]
LPIPS_NET = CONFIG["evaluation"]["lpips_net"]
FIGURES_DIR = PROJECT_ROOT / CONFIG["evaluation"]["figures_dir"]
LOSS_SMOOTHING_WINDOW = CONFIG["evaluation"]["loss_smoothing_window"]

# Smoke test (src/smoke_test.py)
SMOKE_N_ITERATIONS = CONFIG["smoke_test"]["n_iter"]
SMOKE_VISUALIZE_EVERY = CONFIG["smoke_test"]["vis_every"]
SMOKE_OUTPUT_DIR = PROJECT_ROOT / CONFIG["smoke_test"]["output_dir"] / ACTIVE_DATASET  # one folder per dataset
SMOKE_MIN_LOSS_DROP = CONFIG["smoke_test"]["min_loss_drop"]

# All datasets (for scripts that combine results, e.g. report_figures.py)
DATASET_NAMES = list(CONFIG["datasets"].keys())
DATASET_OUTPUT_DIRS = {
    name: PROJECT_ROOT / settings["output_dir"]
    for name, settings in CONFIG["datasets"].items()
}
DATASET_N_ITERATIONS = {name: settings["n_iter"] for name, settings in CONFIG["datasets"].items()}


def _env_bool(name, default):
    """Reads true/false from an environment variable, or returns the config value if it is not set."""
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in ("1", "true", "yes", "on")


# Sampling settings (src/model.py, src/sampling.py, src/render.py, src/train.py)
# Every setting can be overridden with an environment variable, so run_final.sh can queue
# several experiments without editing config.yaml. Not set = value from config.yaml
SAMPLER = os.environ.get("NERF_SAMPLER", CONFIG["sampler"]["type"])                         # uniform | hierarchical | proposal
STRATIFIED = _env_bool("NERF_STRATIFIED", CONFIG["sampler"]["stratified"])
DENSITY_USES_DELTA = _env_bool("NERF_DELTA", CONFIG["sampler"]["density_uses_delta"])       # NeRF eq. 3
N_COARSE = int(os.environ.get("NERF_N_COARSE", CONFIG["sampler"]["n_coarse"]))
N_FINE = int(os.environ.get("NERF_N_FINE", CONFIG["sampler"]["n_fine"]))
N_PTS_PER_RAY = int(os.environ.get("NERF_N_PTS", N_PTS_PER_RAY))                            # uniform runs
SPACING = os.environ.get("NERF_SPACING", CONFIG["sampler"]["spacing"])                    # linear | disparity | lindisp
CONTRACT = _env_bool("NERF_CONTRACT", CONFIG["sampler"]["contract"])
PROPOSAL_HIDDEN = CONFIG["sampler"]["proposal_hidden"]
RAY_MAX_DEPTH = float(os.environ.get("NERF_MAX_DEPTH", RAY_MAX_DEPTH))  # contraction allows a far bound
CONTRACTION_CENTER = [float(v) for v in _DATASET.get("contraction_center", [0.0, 0.0, 0.0])]
CONTRACTION_RADIUS = float(_DATASET.get("contraction_radius", 1.0))
LINDISP_SWITCH = float(_DATASET.get("lindisp_switch", 1.0))
SCENE_BOUND = float(_DATASET.get("scene_bound", 1.5))

# Data fixes (alpha premultiplication for lego, undistortion for poster). Runs with the fixes
# are written to output_dir_fixed / figures_dir_fixed, so they are never mixed with earlier runs
FIX_DATA = _env_bool("NERF_FIX_DATA", CONFIG.get("fix_data", False))
if FIX_DATA:
    OUTPUT_DIR = PROJECT_ROOT / _DATASET["output_dir_fixed"]
    FIGURES_DIR = PROJECT_ROOT / CONFIG["evaluation"]["figures_dir_fixed"]
    DATASET_OUTPUT_DIRS = {
        name: PROJECT_ROOT / settings["output_dir_fixed"]
        for name, settings in CONFIG["datasets"].items()
    }

# Model settings (src/model.py). The defaults in config.yaml give the course book model
ARCHITECTURE = os.environ.get("NERF_ARCH", CONFIG["model"]["architecture"])                # book | nerf | hashgrid
HASHGRID = dict(CONFIG["model"]["hashgrid"])
N_HARMONIC_FUNCTIONS = int(os.environ.get("NERF_N_HARM", N_HARMONIC_FUNCTIONS))
N_HARMONIC_FUNCTIONS_DIR = int(os.environ.get("NERF_N_HARM_DIR", CONFIG["model"]["n_harmonic_functions_dir"]))
HARMONIC_OMEGA0 = float(os.environ.get("NERF_OMEGA0", HARMONIC_OMEGA0))
INCLUDE_INPUT = _env_bool("NERF_INCLUDE_INPUT", CONFIG["model"]["include_input"])
APPEARANCE_DIM = int(os.environ.get("NERF_APPEARANCE_DIM", CONFIG["model"]["appearance_dim"]))

# Training additions
LEARNING_RATE = float(os.environ.get("NERF_LR", LEARNING_RATE))
LR_SCHEDULE = os.environ.get("NERF_LR_SCHEDULE", CONFIG["train"]["lr_schedule"])        # step | exp
LR_FINAL = float(os.environ.get("NERF_LR_FINAL", CONFIG["train"]["lr_final"]))
LR_WARMUP = int(os.environ.get("NERF_LR_WARMUP", CONFIG["train"]["lr_warmup"]))
N_RAYS_PER_IMAGE = int(os.environ.get("NERF_N_RAYS", N_RAYS_PER_IMAGE))
BATCH_SIZE = int(os.environ.get("NERF_BATCH_SIZE", BATCH_SIZE))
DISTORTION_WEIGHT = float(os.environ.get("NERF_DIST_WEIGHT", CONFIG["train"]["distortion_weight"]))
INTERLEVEL_WEIGHT = float(CONFIG["train"]["interlevel_weight"])
ADAM_EPS = float(os.environ.get("NERF_ADAM_EPS", CONFIG["train"]["adam_eps"]))
EVAL_APPEARANCE = os.environ.get("NERF_EVAL_APPEARANCE", CONFIG["evaluation"].get("appearance_code", "mean"))

# One output folder per experiment, so runs don't overwrite each other
EXPERIMENT = os.environ.get("NERF_EXPERIMENT", CONFIG["experiment"])
OUTPUT_DIR = OUTPUT_DIR / EXPERIMENT

# Writes the values actually used back into CONFIG, so train.py saves them in config_used.yaml
CONFIG["experiment"] = EXPERIMENT
CONFIG["active_dataset"] = ACTIVE_DATASET
CONFIG["fix_data"] = FIX_DATA
CONFIG["render"]["n_pts_per_ray"] = N_PTS_PER_RAY
CONFIG["sampler"].update({
    "type": SAMPLER,
    "stratified": STRATIFIED,
    "density_uses_delta": DENSITY_USES_DELTA,
    "n_coarse": N_COARSE,
    "n_fine": N_FINE,
    "spacing": SPACING,
    "contract": CONTRACT,
})
CONFIG["model"].update({
    "architecture": ARCHITECTURE,
    "n_harmonic_functions": N_HARMONIC_FUNCTIONS,
    "n_harmonic_functions_dir": N_HARMONIC_FUNCTIONS_DIR,
    "omega0": HARMONIC_OMEGA0,
    "include_input": INCLUDE_INPUT,
    "appearance_dim": APPEARANCE_DIM,
})
CONFIG["train"].update({
    "learning_rate": LEARNING_RATE,
    "lr_schedule": LR_SCHEDULE,
    "lr_final": LR_FINAL,
    "lr_warmup": LR_WARMUP,
    "batch_size": BATCH_SIZE,
    "adam_eps": ADAM_EPS,
    "distortion_weight": DISTORTION_WEIGHT,
})
CONFIG["render"]["n_rays_per_image"] = N_RAYS_PER_IMAGE
CONFIG["datasets"][ACTIVE_DATASET]["max_depth"] = RAY_MAX_DEPTH


# Quick check that the config loads, prints the main settings of the active dataset
if __name__ == "__main__":
    print(PREVIEW_MESSAGE)
    print(f"Active dataset: {ACTIVE_DATASET}")
    print(f"Experiment:     {EXPERIMENT}")
    print(f"Sampler:        {SAMPLER} (stratified={STRATIFIED}, delta={DENSITY_USES_DELTA}, "
          f"points={N_PTS_PER_RAY if SAMPLER == 'uniform' else f'{N_COARSE}+{N_FINE}'}, "
          f"spacing={SPACING}, contract={CONTRACT})")
    print(f"Model:          {ARCHITECTURE}, L={N_HARMONIC_FUNCTIONS}/{N_HARMONIC_FUNCTIONS_DIR}, "
          f"omega0={HARMONIC_OMEGA0}, include_input={INCLUDE_INPUT}, appearance_dim={APPEARANCE_DIM}")
    print(f"Losses:         lr={LEARNING_RATE} ({LR_SCHEDULE}), distortion_weight={DISTORTION_WEIGHT}, "
          f"rays per step={BATCH_SIZE}x{N_RAYS_PER_IMAGE}")
    print(f"Data dir:       {DATA_DIR}")
    print(f"Output dir:     {OUTPUT_DIR}")
    print(f"Iterations:     {N_ITERATIONS}")