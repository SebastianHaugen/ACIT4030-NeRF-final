# ACIT4030-nerf-final
Improving the PyTorch3D NeRF from the course book (Chapter 6) step by step: better sampling, NeRF positional encoding and MLP, mip-NeRF 360 components (proposal sampling, distortion loss, scene contraction), appearance embeddings, a hash-grid encoding and time as an input for the dynamic lego scene (ACIT4030 final project, OsloMet). Volume rendering is done with PyTorch3D throughout.

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
    NERF_DATASET=lego python -m src.render_video                           # 360° orbit GIFs + filmstrip figure

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
| `model.time_input` (`NERF_TIME`) | The frame time is an extra model input (T-NeRF), for the dynamic lego scene. Test views are rendered at their own time. |
| `evaluation.appearance_code` (`NERF_EVAL_APPEARANCE`) | Code for held-out views: `mean` of all training codes, or `neighbour` (training frames recorded closest in time; results get a `_neighbour` suffix). |
| `train.distortion_weight` (`NERF_DIST_WEIGHT`) | mip-NeRF 360 distortion loss against floaters. |
| `train.lr_schedule: exp`, `lr_final`, `lr_warmup` | Smooth learning-rate decay with warm-up instead of one step. |
| `render.n_rays_per_image` (`NERF_N_RAYS`), `train.adam_eps` | Rays per training step, Adam epsilon (1e-15 for hash grids). |
| `train.grad_accum` (`NERF_GRAD_ACCUM`) | Processes each batch in parts and adds up the gradients: the same update with less GPU memory (used for hierarchical sampling on 8 GB). |

Code: `src/model.py` (fields, encodings, contraction), `src/sampling.py` (sample placement and the mip-NeRF 360 losses), `src/render.py` (rendering used by training, previews and evaluation), `src/load_nerf_data.py` (data loading and fixes).

## Results (held-out views, PSNR / SSIM / LPIPS, training minutes)
Final ablation on the corrected data in `outputs/src_fixed/`. Every run was trained on the same GPU (RTX 3070, 8 GB), so the times are comparable. One seed per run, so differences below about 0.2 dB are within noise. Each "+" row adds one change to the row above it.

| Run | Lego | Poster |
|---|---|---|
| **Sampling** (book model) | | |
| Book model = A2 baseline (`uniform`) | 21.60 / 0.848 / 0.140 (19) | 19.85 / 0.625 / 0.626 (29) |
| + spacing-aware density (`uniform_delta`) | 21.71 / 0.863 / 0.128 (20) | 24.57 / 0.766 / 0.508 (28) |
| + stratified sampling (`uniform_delta_strat`) | 21.63 / 0.860 / 0.131 (18) | 24.49 / 0.768 / 0.511 (28) |
| + 192 instead of 128 samples (`uniform_delta_strat_192`) | 21.75 / 0.865 / 0.126 (27) | 24.67 / 0.773 / 0.502 (41) |
| NeRF hierarchical sampling 64 + 128 (`hier_64_128`) | 21.70 / 0.867 / 0.125 (38) | 25.88 / 0.796 / 0.465 (57) |
| **Model** (from `uniform_delta_strat`) | | |
| + NeRF encoding L = 10 / 4 (`enc_10_4`) | 21.52 / 0.863 / 0.137 (14) | 23.62 / 0.757 / 0.529 (19) |
| + 8-layer NeRF MLP (`enc_mlp`) | 21.94 / 0.870 / 0.132 (37) | 26.15 / 0.823 / 0.417 (53) |
| + distortion loss (`enc_mlp_dist`) | 21.83 / 0.870 / 0.130 (36) | 25.97 / 0.816 / 0.440 (54) |
| + proposal sampling 128 → 64 (`enc_mlp_dist_prop`) | 22.05 / 0.885 / 0.115 (22) | 21.71 / 0.724 / 0.586 (32) |
| + contraction, lin-disp, appearance codes (poster) | | 25.97 / 0.848 / 0.373 (33) |
| + smooth learning rate, 4096 rays | 22.23 / 0.884 / 0.114 (53) | 26.03 / 0.838 / 0.394 (80) |
| + time input (lego) | **23.61** / 0.887 / 0.103 (56) | |
| **Hash grid** (proposal, distortion, 4096 rays; poster also contraction, lin-disp, appearance) | | |
| Hash grid (`hashgrid`) | 21.36 / 0.877 / 0.120 (43) | 24.38 / 0.896 / 0.217 (84) |
| + time input (lego, `hashgrid_time`) | 23.43 / **0.893** / **0.084** (43) | |

Poster runs with appearance codes, evaluated with the codes of the neighbouring training frames instead of the average code (`metrics_neighbour.csv`):

| Run | Poster, neighbour codes |
|---|---|
| + contraction, lin-disp, appearance codes | 27.69 / 0.852 / 0.369 |
| + smooth learning rate, 4096 rays | 27.07 / 0.840 / 0.392 |
| Hash grid | **30.74 / 0.917 / 0.194** |

Notes:
- Best models: lego `hashgrid_time` (best SSIM and LPIPS; `enc_mlp_dist_prop_exp_4096_time` has a slightly higher PSNR), poster `hashgrid` with neighbour codes.
- The provided lego data is the *dynamic* D-NeRF scene (every frame has a `time`, the bucket moves). A static NeRF cannot represent the moving bucket, which caps PSNR at about 22 dB; time as an input lifts it to about 23.5 dB.
- Neighbour codes: poster is a video with automatic exposure, so the average appearance code gives some test views slightly wrong colours. The codes of the training frames recorded just before and after a test frame fix that, using training information only.
- Stratified sampling, more samples and the distortion loss make no real difference here. The spacing-aware density, the NeRF MLP and (for poster) appearance codes and the hash grid do.
- Proposal sampling on poster only works together with contraction and lin-disp spacing; with linear spacing and 64 samples the background gets too few samples.
- 360° orbits of the worst and best model per dataset (GIFs and a filmstrip figure with depth): `outputs/src_fixed/figures/orbits/`.
- `outputs/src/` holds the earlier ablations on the uncorrected data. Their run scripts are in git history (commit `f9ca36c`).

Sampler visualisations: https://docs.nerf.studio/nerfology/model_components/visualize_samplers.html


