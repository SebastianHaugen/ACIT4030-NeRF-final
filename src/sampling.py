"""
Helper functions for placing samples along the rays, and the losses on the ray weights.

- Hierarchical sampling (NeRF, Mildenhall et al. 2020, Section 5.2): the coarse pass tells us
  where along each ray the surface probably is, and the fine pass puts extra samples there.
- Proposal sampling (mip-NeRF 360, Barron et al. 2022, Section 3): a small proposal network
  replaces the coarse NeRF, and is trained with the interlevel loss to bound the NeRF weights.
- Spacing functions (mip-NeRF 360, Section 2): samples evenly spaced in a warped distance s,
  e.g. linear in disparity (1 / depth) so the far background gets fewer samples.
- Distortion loss (mip-NeRF 360, Section 4): pulls the weights on each ray together,
  which removes floaters and semi-transparent "fog".
"""
import torch
from pytorch3d.renderer.implicit.sample_pdf import sample_pdf

SPACINGS = ("linear", "disparity", "lindisp")
PROPOSAL_PADDING = 1e-3  # weight added to every proposal bin before resampling, see sample_proposal_lengths


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


# Spacing functions

def _spacing_fn(t, spacing, switch):
    """The warp g(t) from mip-NeRF 360. Samples are evenly spaced in g(t).

    linear:    g(t) = t                     (the baseline)
    disparity: g(t) = 1 / t                 (evenly spaced in inverse depth)
    lindisp:   g(t) = t / (2 switch)        for t < switch
               g(t) = 1 - switch / (2 t)    for t >= switch
               Linear close to the camera and disparity further away, as in nerfstudio's
               UniformLinDispPiecewiseSampler. g is continuous and smooth at t = switch.
    """
    if spacing == "linear":
        return t
    if spacing == "disparity":
        return 1.0 / t
    if spacing == "lindisp":
        return torch.where(t < switch, t / (2 * switch), 1 - switch / (2 * t))
    raise ValueError(f"Unknown spacing '{spacing}', expected one of {SPACINGS}")


def _spacing_fn_inv(x, spacing, switch):
    """Inverse of _spacing_fn."""
    if spacing == "linear":
        return x
    if spacing == "disparity":
        return 1.0 / x
    if spacing == "lindisp":
        return torch.where(x < 0.5, 2 * switch * x, switch / (2 * (1 - x)))
    raise ValueError(f"Unknown spacing '{spacing}', expected one of {SPACINGS}")


def t_to_s(t, spacing, near, far, switch):
    """Maps depths t in [near, far] to normalised distances s in [0, 1]."""
    g_near = _spacing_fn(torch.as_tensor(near, dtype=t.dtype, device=t.device), spacing, switch)
    g_far = _spacing_fn(torch.as_tensor(far, dtype=t.dtype, device=t.device), spacing, switch)
    return (_spacing_fn(t, spacing, switch) - g_near) / (g_far - g_near)


def s_to_t(s, spacing, near, far, switch):
    """Maps normalised distances s in [0, 1] back to depths t in [near, far]."""
    g_near = _spacing_fn(torch.as_tensor(near, dtype=s.dtype, device=s.device), spacing, switch)
    g_far = _spacing_fn(torch.as_tensor(far, dtype=s.dtype, device=s.device), spacing, switch)
    return _spacing_fn_inv(g_near + s * (g_far - g_near), spacing, switch)


def spaced_lengths(like, spacing, near, far, switch, jitter):
    """Sample depths evenly spaced in s, with the same shape as `like` (..., P).

    jitter: True = one random depth inside each stratum (stratified sampling, training only),
            the same scheme as PyTorch3D's stratified_sampling, but in s instead of t.
    """
    n_pts = like.shape[-1]
    s = torch.linspace(0.0, 1.0, n_pts, dtype=like.dtype, device=like.device).expand(like.shape)
    if jitter:
        mids = 0.5 * (s[..., 1:] + s[..., :-1])
        upper = torch.cat([mids, s[..., -1:]], dim=-1)
        lower = torch.cat([s[..., :1], mids], dim=-1)
        s = lower + (upper - lower) * torch.rand_like(lower)
    return s_to_t(s.contiguous(), spacing, near, far, switch)


# Proposal sampling

def lengths_to_edges(lengths):
    """Interval edges (..., P + 1) for sample depths (..., P).

    Sample i stands for the interval [t_i, t_i+1], and the last sample reuses the previous
    spacing, which is the same convention the model uses for the opacity (density_to_alpha).
    """
    last = lengths[..., -1:] + (lengths[..., -1:] - lengths[..., -2:-1])
    return torch.cat([lengths, last], dim=-1)


def sample_proposal_lengths(lengths, weights, n_samples, deterministic, padding=PROPOSAL_PADDING):
    """Draws n_samples depths per ray from the proposal histogram (mip-NeRF 360).

    Unlike hierarchical sampling, the new samples replace the proposal samples instead of
    being merged with them, so the NeRF model only evaluates n_samples points per ray.

    lengths: (..., P) proposal sample depths, weights: (..., P) proposal weights
    padding: small weight added to every bin (as nerfstudio's histogram_padding). Without it, the
             compiled PyTorch3D sample_pdf puts every sample on the far edge when all weights are
             about zero, which is the case at the start of training, and nothing can be learned.
             It also keeps a few samples spread over the whole ray, so new surfaces can be found.
    returns: (..., n_samples) sorted depths, without gradients
    """
    edges = lengths_to_edges(lengths).detach()
    samples = sample_pdf(edges, weights.detach() + padding, n_samples, det=deterministic)
    # Random samples from sample_pdf are not sorted, but the raymarcher needs them in order
    samples, _ = torch.sort(samples, dim=-1)
    return samples.detach()


def outer_measure(edges, env_edges, env_weights):
    """For every interval in `edges`, the sum of the `env_weights` of all envelope intervals
    that overlap it (mip-NeRF 360, eq. 13). This is the most weight the envelope (proposal)
    histogram can put on that interval.

    edges: (..., N + 1), env_edges: (..., P + 1), env_weights: (..., P)
    returns: (..., N)
    """
    n_env = env_weights.shape[-1]
    cumulative = torch.cat(
        [torch.zeros_like(env_weights[..., :1]), torch.cumsum(env_weights, dim=-1)], dim=-1
    )  # (..., P + 1), weight in front of each envelope edge
    # Index of the first envelope edge that lies after each query edge
    idx = torch.searchsorted(env_edges.contiguous(), edges.contiguous(), right=True)
    idx_lo = (idx - 1).clamp(0, n_env)  # last envelope edge at or before the query edge
    idx_hi = idx.clamp(0, n_env)        # first envelope edge after the query edge
    cum_lo = torch.gather(cumulative, -1, idx_lo)
    cum_hi = torch.gather(cumulative, -1, idx_hi)
    return cum_hi[..., 1:] - cum_lo[..., :-1]


def interlevel_loss(lengths, weights, proposal_lengths, proposal_weights, eps=1e-7):
    """Proposal loss from mip-NeRF 360 (eq. 14).

    Penalises NeRF weight that the proposal histogram does not cover. The NeRF weights and all
    depths are detached, so only the proposal network learns from this loss.
    """
    weights = weights.detach()
    bound = outer_measure(
        lengths_to_edges(lengths).detach(),
        lengths_to_edges(proposal_lengths).detach(),
        proposal_weights,
    )
    loss = (weights - bound).clamp_min(0.0) ** 2 / (weights + eps)
    return loss.sum(dim=-1).mean()


def distortion_loss(s_edges, weights):
    """Distortion loss from mip-NeRF 360 (eq. 15), in linear time.

    s_edges: (..., N + 1) sorted interval edges in normalised distance s
    weights: (..., N) compositing weights of the intervals

    L = sum_ij w_i w_j |m_i - m_j|  +  1/3 sum_i w_i^2 (s_i+1 - s_i),  m = interval midpoints.
    Because the midpoints are sorted, the double sum equals
    2 sum_i w_i (m_i * sum_{j<i} w_j - sum_{j<i} w_j m_j), which needs only cumulative sums.
    """
    mids = 0.5 * (s_edges[..., 1:] + s_edges[..., :-1])
    widths = s_edges[..., 1:] - s_edges[..., :-1]
    w_before = torch.cumsum(weights, dim=-1) - weights
    wm_before = torch.cumsum(weights * mids, dim=-1) - weights * mids
    inter = 2 * (weights * (mids * w_before - wm_before)).sum(dim=-1)
    intra = (weights ** 2 * widths).sum(dim=-1) / 3
    return (inter + intra).mean()