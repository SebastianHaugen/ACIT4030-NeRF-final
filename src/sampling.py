"""
Helper functions for hierarchical sampling (NeRF, Mildenhall et al. 2020, Section 5.2).
The coarse pass tells us where along each ray the surface probably is,
and the fine pass puts extra samples there.
"""
import torch
from pytorch3d.renderer.implicit.sample_pdf import sample_pdf


def weights_from_alpha(alpha):
    """Turns per-sample opacities into compositing weights.

    Same formula as PyTorch3D's EmissionAbsorptionRaymarcher:
    weight_i = alpha_i * product over j < i of (1 - alpha_j)
    i.e. how much sample i contributes to the final pixel colour.

    alpha: (..., P) opacities between 0 and 1
    returns: (..., P) weights
    """
    # Transmittance: how much light gets through all samples before this one
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    # Shift by one so sample i only uses the samples in front of it
    transmittance = torch.cat(
        [torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], dim=-1
    )
    return alpha * transmittance


def sample_fine_lengths(lengths, weights, n_fine, deterministic):
    """Draws n_fine new sample depths per ray, concentrated where the coarse weights are high.

    lengths: (..., P) depths of the coarse samples (sorted)
    weights: (..., P) coarse compositing weights
    n_fine: number of new samples per ray
    deterministic: True at test time (evenly spaced in the CDF), False in training (random)
    returns: (..., n_fine) new depths, without gradients
    """
    # The bins are the midpoints between coarse samples, as in the original NeRF code.
    # P samples give P - 1 midpoints, which form P - 2 bins, one for each inner sample weight
    midpoints = 0.5 * (lengths[..., 1:] + lengths[..., :-1])
    inner_weights = weights[..., 1:-1].detach()  # no gradient through where the samples land

    fine = sample_pdf(midpoints, inner_weights, n_fine, det=deterministic)
    return fine.detach()