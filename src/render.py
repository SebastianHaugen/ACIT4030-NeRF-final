"""
Renders rays with uniform sampling (coarse model only) or hierarchical sampling (coarse + fine).
Used by train.py, the training previews and evaluate.py, so they all render exactly the same way.
"""
import torch

from src.sampling import weights_from_alpha, sample_fine_lengths


def render_rays(ray_bundle, coarse, fine, raymarcher, n_fine, training, chunked=False):
    """Renders a ray bundle and returns (fine_output, coarse_output).

    ray_bundle: PyTorch3D RayBundle from a raysampler (its lengths are the coarse samples)
    coarse:     NeuralRadianceField for the coarse pass
    fine:       NeuralRadianceField for the fine pass, or None for uniform sampling
    raymarcher: EmissionAbsorptionRaymarcher
    n_fine:     extra samples per ray in the fine pass
    training:   True = random fine samples, False = deterministic (evaluation, previews)
    chunked:    True = use batched_forward to save memory (full-image rendering)

    Each output has shape (..., 4): RGB plus opacity (silhouette).
    With uniform sampling the first output is the coarse render and the second is None.
    """
    def run(model, bundle):
        return model.batched_forward(bundle) if chunked else model(bundle)

    # Coarse pass, on the samples the raysampler produced
    densities_c, colors_c = run(coarse, ray_bundle)
    out_coarse = raymarcher(rays_densities=densities_c, rays_features=colors_c)

    if fine is None:
        return out_coarse, None

    # Where along each ray did the coarse pass put its weight?
    weights = weights_from_alpha(densities_c[..., 0].detach())

    # Extra samples where the weights are high, merged with the coarse samples
    fine_lengths = sample_fine_lengths(ray_bundle.lengths, weights, n_fine, deterministic=not training)
    all_lengths, _ = torch.sort(torch.cat([ray_bundle.lengths, fine_lengths], dim=-1), dim=-1)

    # Same rays and pixel positions, only the sample depths change
    fine_bundle = ray_bundle._replace(lengths=all_lengths)

    densities_f, colors_f = run(fine, fine_bundle)
    out_fine = raymarcher(rays_densities=densities_f, rays_features=colors_f)
    return out_fine, out_coarse