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

Each run writes to `outputs/src/<dataset>/<experiment>/`. To run an experiment, set `experiment:` and the `sampler:` block in `src/config.yaml`, then:

    $env:NERF_DATASET="lego"; python -m src.train; python -m src.evaluate

https://docs.nerf.studio/nerfology/model_components/visualize_samplers.html

Read more at NeRF studios: https://docs.nerf.studio/nerfology/model_components/visualize_samplers.html


