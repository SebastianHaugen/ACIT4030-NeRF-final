# ACIT4030-nerf-final
Improving the PyTorch3D NeRF baseline with proposal sampling, scene contraction and appearance embeddings (ACIT4030 final project, OsloMet).

## Sampling
We're adding sampling to improve the final result.

The baseline samples 128 evenly spaced points along every ray, so most samples land in empty space.
`src/` adds the sampling improvements from NeRF (Mildenhall et al., 2020), each switchable in the `sampler:` block of `src/config.yaml`:

| Setting | What it does |
|---|---|
| `density_uses_delta` | Opacity depends on the distance between samples (NeRF eq. 3). Needed once samples are no longer evenly spaced. |
| `stratified` | Randomly jitters the sample depths during training, so the network sees the whole ray instead of the same fixed depths. Off at test time. |
| `type: hierarchical` | Two passes: a coarse model on `n_coarse` samples estimates where the surface is, then `n_fine` extra samples are drawn there (inverse-CDF sampling) and a fine model renders the final colour. |

Code: `src/sampling.py` (weights and fine-sample drawing), `src/render.py` (coarse/fine rendering used by training, previews and evaluation), `src/model.py` (spacing-aware opacity).

## Model and mip-NeRF 360 additions
Also switchable in `src/config.yaml` (the defaults give the book model), or with the environment variable in brackets:

| Setting | What it does |
|---|---|
| `model.n_harmonic_functions(_dir)`, `omega0`, `include_input` (`NERF_N_HARM`, `NERF_N_HARM_DIR`, `NERF_OMEGA0`, `NERF_INCLUDE_INPUT`) | NeRF positional encoding: L = 10 / 4 starting at pi, instead of 60 frequencies starting at 0.1 (most of which are float32 noise). |
| `model.architecture: nerf` (`NERF_ARCH`) | 8-layer ReLU MLP with a skip connection, as in the NeRF paper. |
| `sampler.type: proposal` (`NERF_SAMPLER`) | mip-NeRF 360 proposal sampling: a small density-only network on `n_coarse` samples, trained with the interlevel loss, decides where the NeRF model evaluates its `n_fine` samples. |
| `train.distortion_weight` (`NERF_DIST_WEIGHT`) | mip-NeRF 360 distortion loss, pulls the weights on each ray together (fewer floaters). |
| `sampler.spacing: lindisp` (`NERF_SPACING`) | Samples linear in depth up to `lindisp_switch`, then linear in disparity, so far-away space gets fewer samples. |
| `sampler.contract` (`NERF_CONTRACT`) | mip-NeRF 360 scene contraction: points outside `contraction_radius` are squeezed into a shell, so a far `max_depth` (`NERF_MAX_DEPTH`) can be used. |
| `model.appearance_dim` (`NERF_APPEARANCE_DIM`) | NeRF-W appearance code per training image (exposure / white balance). Test views use the mean code. |

`run_improvements.ps1 -Dataset lego|poster` runs these as an ablation, one change at a time.

Each run writes to `outputs/src/<dataset>/<experiment>/`. To run an experiment, set `experiment:` and the `sampler:` block in `src/config.yaml`, then:

    $env:NERF_DATASET="lego"; python -m src.train; python -m src.evaluate

https://docs.nerf.studio/nerfology/model_components/visualize_samplers.html

Read more at NeRF studios: https://docs.nerf.studio/nerfology/model_components/visualize_samplers.html


