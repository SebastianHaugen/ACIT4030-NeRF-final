# ACIT4030-nerf-final
Improving the PyTorch3D NeRF from the course book (Chapter 6) step by step: NeRF positional encoding and MLP, proposal sampling, scene contraction, appearance embeddings and a hash-grid encoding (ACIT4030 final project, OsloMet).

## Project structure
| Folder | Contents |
|---|---|
| `baseline/` | The A2 baseline (course book model), frozen. Own `config.yaml`, results in `outputs/baseline/`. |
| `src/` | The final project. Every addition is a setting in `src/config.yaml`; the defaults give the book model. |
| `notebooks/` | Data check and the Colab training notebook. |
| `data/` | `lego` and `poster` (not in git). |
| `outputs/` | Results: metrics, loss histories, configs, test renders and figures. Checkpoints and previews are not in git. |
| `report/figures/` | Figures used in the report. |

## Setup
    pip install -r requirements.txt

PyTorch and PyTorch3D are installed separately, matching your CUDA version (see `requirements.txt`).

## Running
Single experiment (settings from `src/config.yaml`, any of them can be overridden with a `NERF_*` environment variable, see `src/repo_util/LoadConfigurations.py`):

    NERF_DATASET=poster NERF_EXPERIMENT=my_run python -m src.smoke_test   # 1000-iteration check
    NERF_DATASET=poster NERF_EXPERIMENT=my_run python -m src.train
    NERF_DATASET=poster NERF_EXPERIMENT=my_run python -m src.evaluate
    python -m src.report_figures                                           # tables + loss plots

Full final ablation (skips runs that are already finished):

    PYTHON=/path/to/python ./run_final.sh           # Linux (add --smoke for smoke tests only)
    powershell -ExecutionPolicy Bypass -File run_final.ps1   # Windows

## What `src/` adds
| Setting (environment variable) | What it does |
|---|---|
| `fix_data` (`NERF_FIX_DATA`) | Data fixes: lego colours are multiplied by alpha (the renderer composites onto black), poster images are undistorted with square pixels (the loader ignored the lens distortion and `fl_x` ≠ `fl_y`). |
| `sampler.density_uses_delta` (`NERF_DELTA`) | Opacity depends on the distance between samples (NeRF eq. 3). Needed once samples are not evenly spaced. |
| `sampler.stratified` (`NERF_STRATIFIED`) | Random jitter of the sample depths during training. |
| `sampler.type: hierarchical` | NeRF coarse + fine sampling with two networks. |
| `sampler.type: proposal` (`NERF_SAMPLER`) | mip-NeRF 360 proposal sampling: a small density-only network (trained with the interlevel loss) decides where the NeRF model samples. |
| `sampler.spacing: lindisp` (`NERF_SPACING`) | Samples linear in depth up to `lindisp_switch`, then linear in disparity. |
| `sampler.contract` (`NERF_CONTRACT`) | mip-NeRF 360 scene contraction, so a far `max_depth` (`NERF_MAX_DEPTH`) can be used. |
| `model.n_harmonic_functions(_dir)`, `omega0`, `include_input` | NeRF positional encoding (L = 10 / 4 from pi) instead of 60 frequencies from 0.1. |
| `model.architecture` (`NERF_ARCH`) | `book` (2 layers), `nerf` (8 layers + skip) or `hashgrid` (Instant-NGP hash encoding + tiny MLPs, set up like nerfacto, plain PyTorch). |
| `model.appearance_dim` (`NERF_APPEARANCE_DIM`) | NeRF-W appearance code per training image (exposure / white balance). |
| `evaluation.appearance_code` (`NERF_EVAL_APPEARANCE`) | Code for held-out views: `mean` of all training codes, or `neighbour` (training frames recorded closest in time; results get a `_neighbour` suffix). |
| `train.distortion_weight` (`NERF_DIST_WEIGHT`) | mip-NeRF 360 distortion loss against floaters. |
| `train.lr_schedule: exp`, `lr_final`, `lr_warmup` | Smooth learning-rate decay with warm-up instead of one step. |
| `render.n_rays_per_image` (`NERF_N_RAYS`), `train.adam_eps` | Rays per training step, Adam epsilon (1e-15 for hash grids). |

Code: `src/model.py` (fields, encodings, contraction), `src/sampling.py` (sample placement and the mip-NeRF 360 losses), `src/render.py` (rendering used by training, previews and evaluation), `src/load_nerf_data.py` (data loading and fixes).

## Results (held-out views, PSNR / SSIM / LPIPS)
Final ablation on the corrected data, `outputs/src_fixed/` (one seed per run):

| Run | Lego | Poster |
|---|---|---|
| Book model (`uniform`) | 21.60 / 0.848 / 0.140 | 19.85 / 0.625 / 0.626 |
| + spacing-aware density | 21.71 / 0.863 / 0.128 | 24.57 / 0.766 / 0.508 |
| + NeRF encoding and MLP | 21.94 / 0.870 / 0.132 | 26.15 / 0.823 / 0.417 |
| + proposal sampling, distortion loss | 22.05 / 0.885 / 0.115 | 21.71 / 0.724 / 0.586 |
| + contraction, lin-disp, appearance (poster) | | 25.97 / 0.848 / 0.373 (neighbour: 27.69 / 0.852 / 0.369) |
| + smooth LR, 4096 rays | 22.23 / 0.884 / 0.114 | 26.03 / 0.838 / 0.394 (neighbour: 27.07 / 0.840 / 0.392) |
| Hash grid (all additions) | 21.36 / 0.877 / 0.120 | 24.38 / 0.896 / 0.217 (**neighbour: 30.74 / 0.917 / 0.194**) |

Notes:
- The provided lego data is the *dynamic* D-NeRF scene (every frame has a `time`, the bucket moves). A static NeRF cannot represent the moving bucket, which caps PSNR at about 22 dB for every model.
- Proposal sampling on poster only works together with contraction and lin-disp spacing; with linear spacing and 64 samples the background gets too few samples.
- `outputs/src/` holds the earlier sampling and model ablations on the uncorrected data. Their run scripts are in git history (commit `f9ca36c`).

Sampler visualisations: https://docs.nerf.studio/nerfology/model_components/visualize_samplers.html


