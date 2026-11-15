"""
Renders rays with uniform sampling (one model), hierarchical sampling (coarse + fine NeRF)
or proposal sampling (proposal network + NeRF).
Used by train.py, the training previews and evaluate.py, so they all render exactly the same way.
"""
from dataclasses import dataclass

import torch

from src.sampling import (
    weights_from_alpha,
    sample_fine_lengths,
    sample_proposal_lengths,
    spaced_lengths,
)

SAMPLERS = ("uniform", "hierarchical", "proposal")


@dataclass
class SamplerSettings:
    """Everything render_rays needs to know about where to put the samples. Saved in the checkpoint."""
    type: str            # uniform | hierarchical | proposal
    n_first: int         # samples per ray in the first (or only) pass
    n_fine: int          # hierarchical: extra samples, proposal: samples for the NeRF model
    min_depth: float
    max_depth: float
    spacing: str = "linear"       # linear | disparity | lindisp (see sampling.py)
    lindisp_switch: float = 1.0
    stratified: bool = False


def render_rays(ray_bundle, first, second, raymarcher, settings, training, chunked=False,
                appearance_codes=None, times=None):
    """Renders a ray bundle.

    ray_bundle: PyTorch3D RayBundle from a raysampler (linearly spaced lengths)
    first:      uniform: the NeRF model, hierarchical: the coarse NeRF, proposal: the ProposalField
    second:     hierarchical: the fine NeRF, proposal: the NeRF model, uniform: None
    raymarcher: EmissionAbsorptionRaymarcher
    settings:   SamplerSettings
    training:   True = random samples (if stratified), False = deterministic (evaluation, previews)
    chunked:    True = use batched_forward to save memory (full-image rendering)
    appearance_codes: optional (B, appearance_dim), one code per camera in the batch
    times:      optional (B,), the time of each camera's frame (dynamic scenes, time_input models).
                Only the NeRF models get it; the proposal network stays static and learns where
                the scene can be at any time, which is enough to place the samples

    Returns a dict:
        rgba:             (..., 4) final render, RGB plus opacity (silhouette)
        rgba_coarse:      (..., 4) coarse render (hierarchical only, else None)
        weights, lengths: (..., P) compositing weights and depths of the final samples
        proposal_weights, proposal_lengths: the same for the proposal pass (proposal only, else None)
    """
    # Samples spaced in a warped distance replace PyTorch3D's linearly spaced ones.
    # Linear spacing keeps PyTorch3D's samples, so the earlier experiments are reproduced exactly
    if settings.spacing != "linear":
        lengths = spaced_lengths(
            ray_bundle.lengths, settings.spacing, settings.min_depth, settings.max_depth,
            settings.lindisp_switch, jitter=training and settings.stratified,
        )
        ray_bundle = ray_bundle._replace(lengths=lengths)

    # One appearance code and one time per camera, repeated for every ray of that camera
    ray_shape = ray_bundle.origins.shape[:-1]
    if appearance_codes is not None:
        view_shape = (appearance_codes.shape[0],) + (1,) * (len(ray_shape) - 1) + (-1,)
        appearance_codes = appearance_codes.view(view_shape).expand(*ray_shape, -1)
    if times is not None:
        view_shape = (times.shape[0],) + (1,) * (len(ray_shape) - 1) + (1,)
        times = times.reshape(view_shape).expand(*ray_shape, 1)

    def run(model, bundle):
        if chunked:
            return model.batched_forward(bundle, appearance_codes=appearance_codes, times=times)
        return model(bundle, appearance_codes=appearance_codes, times=times)

    out = {"rgba_coarse": None, "proposal_weights": None, "proposal_lengths": None}

    if settings.type == "uniform":
        final_model, final_bundle = first, ray_bundle

    elif settings.type == "hierarchical":
        # Coarse pass, on the first samples
        densities_c, colors_c = run(first, ray_bundle)
        out["rgba_coarse"] = raymarcher(rays_densities=densities_c, rays_features=colors_c)

        # Where along each ray did the coarse pass put its weight?
        weights_c = weights_from_alpha(densities_c[..., 0].detach())

        # Extra samples where the weights are high, merged with the coarse samples
        fine_lengths = sample_fine_lengths(ray_bundle.lengths, weights_c, settings.n_fine,
                                           deterministic=not training)
        all_lengths, _ = torch.sort(torch.cat([ray_bundle.lengths, fine_lengths], dim=-1), dim=-1)

        # Same rays and pixel positions, only the sample depths change
        final_model, final_bundle = second, ray_bundle._replace(lengths=all_lengths)

    elif settings.type == "proposal":
        # Proposal pass: density only, no colour
        densities_p = first.batched_forward(ray_bundle) if chunked else first(ray_bundle)
        weights_p = weights_from_alpha(densities_p[..., 0])
        out["proposal_weights"], out["proposal_lengths"] = weights_p, ray_bundle.lengths

        # The NeRF model only gets the samples drawn from the proposal histogram
        nerf_lengths = sample_proposal_lengths(ray_bundle.lengths, weights_p, settings.n_fine,
                                               deterministic=not training)
        final_model, final_bundle = second, ray_bundle._replace(lengths=nerf_lengths)

    else:
        raise ValueError(f"Unknown sampler type '{settings.type}', expected one of {SAMPLERS}")

    densities, colors = run(final_model, final_bundle)
    out["rgba"] = raymarcher(rays_densities=densities, rays_features=colors)
    out["weights"] = weights_from_alpha(densities[..., 0])
    out["lengths"] = final_bundle.lengths
    return out