import math

import torch
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points

from src.repo_util.LoadConfigurations import (
    N_HARMONIC_FUNCTIONS,
    N_HARMONIC_FUNCTIONS_DIR,
    HARMONIC_OMEGA0,
    INCLUDE_INPUT,
    N_HIDDEN_NEURONS,
    N_BATCHES_FULL_RENDER,
    DENSITY_USES_DELTA,
    ARCHITECTURE,
    APPEARANCE_DIM,
    CONTRACT,
    CONTRACTION_CENTER,
    CONTRACTION_RADIUS,
    SCENE_BOUND,
    HASHGRID,
    PROPOSAL_HIDDEN,
)

"""
Baseline model from the course book (Chapter 6), with these changes for the final project.
Each one is a setting in config.yaml, and the defaults give the original book model:
- density_uses_delta: the opacity takes the spacing between samples into account (NeRF eq. 3).
  Needed whenever the samples along a ray are not evenly spaced (hierarchical, proposal, lindisp).
- n_harmonic_functions(_dir), omega0, include_input: positional encoding as in NeRF (L = 10 and 4).
- architecture: nerf = 8 ReLU layers with a skip connection (NeRF, Mildenhall et al. 2020, Fig. 7),
  hashgrid = multiresolution hash encoding with a tiny MLP (Instant-NGP, Mueller et al. 2022),
  set up like nerfstudio's nerfacto.
- contract: scene contraction from mip-NeRF 360 (Barron et al. 2022, eq. 10).
- appearance_dim: per-image appearance code for the colour head (NeRF in the Wild, Martin-Brualla et al. 2021).
ProposalField is the small density-only network for proposal sampling (mip-NeRF 360, Sec. 3).
"""


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=N_HARMONIC_FUNCTIONS, omega0=HARMONIC_OMEGA0,
                 include_input=False, persistent=True):
        """Turns each input coordinate into sine and cosine values at increasing frequencies."""
        super().__init__()
        self.include_input = include_input
        # Frequencies are omega0 times 1, 2, 4, 8 and so on
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
            persistent=persistent,
        )

    def output_dim(self, input_dim=3):
        """Number of values the embedding returns for an input with input_dim coordinates."""
        return input_dim * (2 * len(self.frequencies) + int(self.include_input))

    def forward(self, x):
        """Returns the embedding of x, with n_harmonic_functions * 2 (+ 1) values per coordinate."""
        # Multiplies every coordinate by every frequency, then flattens them into one vector
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        embed = torch.cat((embed.sin(), embed.cos()), dim=-1)
        if self.include_input:
            # NeRF also passes the raw coordinates, which gives the network the lowest frequency
            embed = torch.cat((embed, x), dim=-1)
        return embed


def contract(points, center, radius):
    """Scene contraction from mip-NeRF 360 (eq. 10), in world units.

    Points within `radius` of `center` are left unchanged. Points further away are pulled into
    a shell between radius and 2 * radius, so the whole unbounded scene fits in a ball and
    the far background still gets a sensible input to the positional encoding.
    """
    x = (points - center) / radius
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x = torch.where(norm <= 1.0, x, (2.0 - 1.0 / norm) * x / norm)
    return center + radius * x


def normalize_points(points, contracted, center, radius, scene_bound):
    """Maps sample points into the unit cube [0, 1]^3 that the hash grid covers.

    contracted: the points went through contract(), so they lie within 2 * radius of the centre.
    Otherwise the grid covers the box center +- scene_bound, and points outside it get zero
    density (the returned mask is 0 for them), as in Instant-NGP for bounded scenes like lego.
    """
    if contracted:
        x = (points - center) / (4 * radius) + 0.5
        inside = torch.ones_like(x[..., :1])
    else:
        x = (points - center) / (2 * scene_bound) + 0.5
        inside = ((x >= 0) & (x <= 1)).all(dim=-1, keepdim=True).to(x.dtype)
    return x.clamp(0.0, 1.0 - 1e-6), inside


class _TruncExp(torch.autograd.Function):
    """exp() with the input clamped in the backward pass (Instant-NGP), so a large density does not
    give exploding gradients. The forward pass is also capped, so density * 0 spacing is never nan."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x.clamp(max=15.0))

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * torch.exp(x.clamp(-15.0, 15.0))


trunc_exp = _TruncExp.apply


class HashEncoding(torch.nn.Module):
    """Multiresolution hash encoding (Instant-NGP, Mueller et al. 2022), in plain PyTorch.

    The unit cube is covered by n_levels grids, from base_res to max_res cells per side.
    Every grid corner has a small learned feature vector; coarse levels store all corners,
    fine levels share a table of 2^log2_table_size entries through a spatial hash. A point gets
    the trilinear interpolation of its 8 surrounding corners on every level, concatenated.
    Most of the scene detail is stored in these tables, so only a tiny MLP is needed after them,
    which is why training is much faster than with the 8-layer NeRF MLP.
    """
    PRIMES = (1, 2654435761, 805459861)  # the hash primes from the Instant-NGP paper

    def __init__(self, n_levels=16, n_features=2, log2_table_size=19, base_res=16, max_res=2048):
        super().__init__()
        growth = math.exp((math.log(max_res) - math.log(base_res)) / max(1, n_levels - 1))
        self.resolutions = [int(math.floor(base_res * growth ** level)) for level in range(n_levels)]
        self.table_size = 2 ** log2_table_size
        # Small uniform initialisation, as in Instant-NGP
        self.table = torch.nn.Parameter(
            torch.empty(n_levels, self.table_size, n_features).uniform_(-1e-4, 1e-4)
        )
        corners = torch.tensor([[(i >> d) & 1 for d in range(3)] for i in range(8)])
        self.register_buffer("corner_offsets", corners, persistent=False)  # (8, 3)
        self.register_buffer("primes", torch.tensor(self.PRIMES), persistent=False)

    def output_dim(self):
        return self.table.shape[0] * self.table.shape[2]

    def forward(self, x):
        """x: (..., 3) in [0, 1). Returns (..., n_levels * n_features)."""
        shape = x.shape[:-1]
        x = x.reshape(-1, 3)
        offsets = self.corner_offsets
        features = []
        for level, res in enumerate(self.resolutions):
            pos = x * res
            cell = torch.floor(pos)
            frac = pos - cell                                   # (N, 3) position inside the cell
            corners = cell.long()[:, None, :] + offsets          # (N, 8, 3) integer corner coordinates
            if (res + 1) ** 3 <= self.table_size:
                # Coarse level: every corner gets its own entry, no collisions
                idx = corners[..., 0] + (res + 1) * (corners[..., 1] + (res + 1) * corners[..., 2])
            else:
                # Fine level: spatial hash, XOR of the coordinates times large primes
                h = corners * self.primes
                idx = (h[..., 0] ^ h[..., 1] ^ h[..., 2]) & (self.table_size - 1)
            # Trilinear weights: frac for the upper corner, 1 - frac for the lower one, per axis
            weights = torch.where(offsets.bool(), frac[:, None, :], 1 - frac[:, None, :]).prod(dim=-1)
            features.append((weights[..., None] * self.table[level][idx]).sum(dim=1))
        return torch.cat(features, dim=-1).view(*shape, -1)


def density_to_alpha(raw_densities, lengths, directions, uses_delta):
    """Maps the raw density of each sample to an opacity between 0 and 1.

    raw_densities: (..., P, 1)  output of the density head
    lengths:       (..., P)     distance of each sample along its ray
    directions:    (..., 3)     ray directions (not unit length in PyTorch3D)
    """
    if not uses_delta:
        # Original book formula: opacity does not depend on the sample spacing
        return 1 - (-raw_densities).exp()

    # NeRF eq. 3: alpha = 1 - exp(-sigma * delta), delta = distance to the next sample
    deltas = lengths[..., 1:] - lengths[..., :-1]
    # The last sample reuses the previous spacing. Original NeRF uses 1e10 here, which makes
    # every ray end fully opaque and would conflict with the silhouette loss on lego
    deltas = torch.cat([deltas, deltas[..., -1:]], dim=-1)  # (..., P)
    # Lengths are measured in units of the direction vector, so scale by its length
    deltas = deltas * directions.norm(dim=-1, keepdim=True)
    return 1 - torch.exp(-raw_densities * deltas[..., None])


MAX_POINTS_PER_CHUNK = 2 ** 18  # sample points per forward pass when rendering a full image


def chunked_forward(forward, ray_bundle, n_batches, **per_ray):
    """Runs forward(bundle, **per_ray_chunks) on chunks of rays, so a full image fits in memory.

    per_ray: optional tensors (..., D) with one value per ray (e.g. appearance codes, times);
             None values are passed on as None
    Returns the outputs of forward joined again, each with shape (..., P, channels).
    """
    n_pts_per_ray = ray_bundle.lengths.shape[-1]
    spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

    # Splits all rays into at least n_batches chunks, and more if a chunk would have more than
    # MAX_POINTS_PER_CHUNK points (a 270x480 poster image at 128 points per ray does not fit in 8 GB)
    tot_samples = ray_bundle.origins.shape[:-1].numel()
    n_batches = max(n_batches, -(-tot_samples * n_pts_per_ray // MAX_POINTS_PER_CHUNK))
    batches = torch.chunk(torch.arange(tot_samples), n_batches)

    # reshape instead of view, since lengths from the fine pass may not be contiguous
    origins = ray_bundle.origins.reshape(-1, 3)
    directions = ray_bundle.directions.reshape(-1, 3)
    lengths = ray_bundle.lengths.reshape(-1, n_pts_per_ray)
    per_ray = {name: (value.reshape(tot_samples, -1) if value is not None else None)
               for name, value in per_ray.items()}

    # Runs the normal forward pass on one chunk at a time
    batch_outputs = [
        forward(
            RayBundle(
                origins=origins[batch_idx],
                directions=directions[batch_idx],
                lengths=lengths[batch_idx],
                xys=None,
            ),
            **{name: (value[batch_idx] if value is not None else None) for name, value in per_ray.items()},
        ) for batch_idx in batches
    ]

    # Joins the chunks again and restores the original shape
    return [
        torch.cat(
            [batch_output[output_i] for batch_output in batch_outputs], dim=0
        ).view(*spatial_size, -1) for output_i in range(len(batch_outputs[0]))
    ]


class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions=N_HARMONIC_FUNCTIONS,
        n_hidden_neurons=N_HIDDEN_NEURONS,
        omega0=HARMONIC_OMEGA0,
        density_uses_delta=DENSITY_USES_DELTA,
        n_harmonic_functions_dir=N_HARMONIC_FUNCTIONS_DIR,
        include_input=INCLUDE_INPUT,
        architecture=ARCHITECTURE,
        appearance_dim=APPEARANCE_DIM,
        contract=CONTRACT,
        contraction_center=CONTRACTION_CENTER,
        contraction_radius=CONTRACTION_RADIUS,
        scene_bound=SCENE_BOUND,
        hashgrid=None,
        time_input=False,
        n_harmonic_functions_time=6,
    ):
        """Builds the MLP that predicts opacity and colour for each sample point.

        time_input: the field also gets the time of the frame (T-NeRF, the time-conditioned baseline
        of D-NeRF, Pumarola et al. 2021), for dynamic scenes such as the D-NeRF lego, where the
        bucket moves. The encoded time is fed in next to the encoded position, so density and
        colour can change over time.
        """
        super().__init__()
        if architecture not in ("book", "nerf", "hashgrid"):
            raise ValueError(f"Unknown architecture '{architecture}', expected 'book', 'nerf' or 'hashgrid'")
        if architecture == "hashgrid" and not density_uses_delta:
            raise ValueError("The hash grid predicts a density per unit length, so it needs density_uses_delta")
        self.density_uses_delta = density_uses_delta
        self.architecture = architecture
        self.appearance_dim = appearance_dim
        self.contract = contract
        self.contraction_radius = contraction_radius
        self.scene_bound = scene_bound
        self.register_buffer(
            "contraction_center", torch.tensor(contraction_center, dtype=torch.float32), persistent=False
        )

        # Encodings for the point positions and the viewing directions. The direction encoding is
        # not saved in the checkpoint (it has no weights), so checkpoints from before still load
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions, omega0, include_input)
        self.harmonic_embedding_dir = HarmonicEmbedding(
            n_harmonic_functions_dir, omega0, include_input, persistent=False
        )
        embedding_dim = self.harmonic_embedding.output_dim()
        # The colour head also gets the embedded direction and the appearance code
        color_extra_dim = self.harmonic_embedding_dir.output_dim() + appearance_dim

        # Time t in [0, 1]: sin/cos at pi * 1, 2, 4, ... plus t itself. Added to the position input
        self.time_input = time_input
        time_dim = 0
        if time_input:
            self.harmonic_embedding_time = HarmonicEmbedding(
                n_harmonic_functions_time, math.pi, include_input=True, persistent=False
            )
            time_dim = self.harmonic_embedding_time.output_dim(input_dim=1)
        embedding_dim += time_dim  # book / nerf: the time is part of the MLP input (and the skip)

        if architecture == "book":
            # Shared part with two hidden layers, Softplus is used instead of ReLU
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, n_hidden_neurons),
                torch.nn.Softplus(beta=10.0),
                torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
                torch.nn.Softplus(beta=10.0),
            )

            # Colour head, takes the shared features together with the embedded viewing direction
            self.color_layer = torch.nn.Sequential(
                torch.nn.Linear(n_hidden_neurons + color_extra_dim, n_hidden_neurons),
                torch.nn.Softplus(beta=10.0),
                torch.nn.Linear(n_hidden_neurons, 3),
                torch.nn.Sigmoid(),
                # Sigmoid keeps the RGB values between 0 and 1
            )
        elif architecture == "hashgrid":
            # nerfacto-style field: hash encoding -> tiny density MLP that also outputs 15 geometry
            # features -> tiny colour MLP that adds the viewing direction (and appearance code)
            hg = {**HASHGRID, **(hashgrid or {})}
            self.encoding = HashEncoding(hg["levels"], hg["features"], hg["log2_table_size"],
                                         hg["base_res"], hg["max_res"])
            n_geo = 15
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.encoding.output_dim() + time_dim, hg["hidden"]),
                torch.nn.ReLU(),
                torch.nn.Linear(hg["hidden"], 1 + n_geo),
            )
            self.color_layer = torch.nn.Sequential(
                torch.nn.Linear(n_geo + color_extra_dim, hg["hidden"]),
                torch.nn.ReLU(),
                torch.nn.Linear(hg["hidden"], hg["hidden"]),
                torch.nn.ReLU(),
                torch.nn.Linear(hg["hidden"], 3),
                torch.nn.Sigmoid(),
            )
            return  # the density comes from self.mlp, so there is no separate density head
        else:
            # NeRF Fig. 7: 8 ReLU layers, the encoded position is fed in again before the 6th layer
            self.skip_at = 5
            self.mlp = torch.nn.ModuleList([
                torch.nn.Linear(
                    embedding_dim if i == 0 else n_hidden_neurons + (embedding_dim if i == self.skip_at else 0),
                    n_hidden_neurons,
                )
                for i in range(8)
            ])
            # Feature vector for the colour head (no activation, as in NeRF)
            self.feature_layer = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
            # One hidden layer of half the width for the view-dependent colour
            self.color_layer = torch.nn.Sequential(
                torch.nn.Linear(n_hidden_neurons + color_extra_dim, n_hidden_neurons // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(n_hidden_neurons // 2, 3),
                torch.nn.Sigmoid(),
            )

        # Density head, depends on the position only
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, 1),
            torch.nn.Softplus(beta=10.0),
            # Softplus keeps the raw density positive
        )

        # A bias of minus 1.5 makes every point start almost transparent
        # The book notes this is important for the model to converge.
        # With density_uses_delta the raw density is multiplied by the (small) sample spacing,
        # so if training starts out empty for too long, try a higher value here (e.g. 0.0)
        self.density_layer[0].bias.data[0] = -1.5

    def _ray_points(self, ray_bundle):
        """Turns ray origins, directions and lengths into 3D sample points, contracted if enabled."""
        points = ray_bundle_to_ray_points(ray_bundle)
        if self.contract:
            points = contract(points, self.contraction_center, self.contraction_radius)
        return points

    def _trunk(self, embeds):
        """Returns the features for the density head and the features for the colour head."""
        if self.architecture == "book":
            # Shared features for both heads
            features = self.mlp(embeds)
            return features, features
        h = embeds
        for i, layer in enumerate(self.mlp):
            if i == self.skip_at:
                h = torch.cat((h, embeds), dim=-1)
            h = torch.relu(layer(h))
        return h, self.feature_layer(h)

    def _get_colors(self, features, rays_directions, appearance_codes=None):
        """Predicts the RGB colour of each point from its features, the viewing direction
        and (if enabled) the appearance code of the image the ray belongs to."""
        spatial_size = features.shape[:-1]

        # Scales the directions to unit length
        rays_directions_normed = torch.nn.functional.normalize(
            rays_directions, dim=-1
        )

        per_ray = [self.harmonic_embedding_dir(rays_directions_normed)]
        if self.appearance_dim:
            if appearance_codes is None:
                appearance_codes = rays_directions.new_zeros(*rays_directions.shape[:-1], self.appearance_dim)
            per_ray.append(appearance_codes)
        per_ray = torch.cat(per_ray, dim=-1)

        # Every point on a ray shares the same direction, so the embedding is repeated per point
        per_ray_expand = per_ray[..., None, :].expand(
            *spatial_size, per_ray.shape[-1]
        )

        color_layer_input = torch.cat(
            (features, per_ray_expand),
            dim=-1
        )
        return self.color_layer(color_layer_input)

    def forward(
        self,
        ray_bundle: RayBundle,
        appearance_codes=None,
        times=None,
        **kwargs,
    ):
        """Returns the opacity and RGB colour of every sample point along the rays.

        appearance_codes: optional (..., appearance_dim), one code per ray
        times:            (..., 1) time of the frame each ray belongs to (only with time_input)
        """
        rays_points_world = self._ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]

        # Encoded time, repeated for every point on the ray
        time_features = None
        if self.time_input:
            if times is None:
                raise ValueError("This model was trained with time_input, so it needs the frame times")
            time_features = self.harmonic_embedding_time(times)
            time_features = time_features[..., None, :].expand(*rays_points_world.shape[:-1], -1)

        if self.architecture == "hashgrid":
            x, inside = normalize_points(rays_points_world, self.contract, self.contraction_center,
                                         self.contraction_radius, self.scene_bound)
            features = self.encoding(x)
            if time_features is not None:
                features = torch.cat((features, time_features), dim=-1)
            h = self.mlp(features)
            # Density per unit length, exp activation as in Instant-NGP / nerfacto
            raw_densities = trunc_exp(h[..., :1] - 1.0) * inside
            color_features = h[..., 1:]
        else:
            embeds = self.harmonic_embedding(
                rays_points_world
            )
            if time_features is not None:
                embeds = torch.cat((embeds, time_features), dim=-1)
            # embeds.shape = [minibatch x ... x embedding_dim]

            density_features, color_features = self._trunk(embeds)
            # features.shape = [minibatch x ... x n_hidden_neurons]

            raw_densities = self.density_layer(density_features)
        rays_densities = density_to_alpha(
            raw_densities, ray_bundle.lengths, ray_bundle.directions, self.density_uses_delta
        )
        # rays_densities.shape = [minibatch x ... x 1]

        rays_colors = self._get_colors(color_features, ray_bundle.directions, appearance_codes)
        # rays_colors.shape = [minibatch x ... x 3]

        return rays_densities, rays_colors

    def batched_forward(
        self,
        ray_bundle: RayBundle,
        appearance_codes=None,
        times=None,
        n_batches: int = N_BATCHES_FULL_RENDER,
        **kwargs,
    ):
        """Runs forward in chunks so a full image can be rendered without running out of memory."""
        rays_densities, rays_colors = chunked_forward(
            self.forward, ray_bundle, n_batches, appearance_codes=appearance_codes, times=times,
        )
        return rays_densities, rays_colors


class ProposalField(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions=N_HARMONIC_FUNCTIONS,
        omega0=HARMONIC_OMEGA0,
        include_input=INCLUDE_INPUT,
        n_hidden_neurons=PROPOSAL_HIDDEN,
        contract=CONTRACT,
        contraction_center=CONTRACTION_CENTER,
        contraction_radius=CONTRACTION_RADIUS,
        architecture="mlp",
        scene_bound=SCENE_BOUND,
    ):
        """Small density-only network for proposal sampling (mip-NeRF 360, Sec. 3).

        It never predicts colour. It is only trained to put its weight where the NeRF model
        has its weight (interlevel loss), so it can tell the NeRF model where to sample.
        Samples are not evenly spaced, so the opacity always uses the sample spacing.
        architecture: mlp (positional encoding + small MLP) or hashgrid (a small hash grid with
        5 levels up to 128 cells and a 16-neuron MLP, the proposal network of nerfacto).
        """
        super().__init__()
        self.contract = contract
        self.contraction_radius = contraction_radius
        self.architecture = architecture
        self.scene_bound = scene_bound
        self.register_buffer(
            "contraction_center", torch.tensor(contraction_center, dtype=torch.float32), persistent=False
        )
        if architecture == "hashgrid":
            self.encoding = HashEncoding(n_levels=5, n_features=2, log2_table_size=17, base_res=16, max_res=128)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.encoding.output_dim(), 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
            )
            return
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions, omega0, include_input)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.harmonic_embedding.output_dim(), n_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.ReLU(),
        )
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, 1),
            torch.nn.Softplus(beta=10.0),
        )
        # Starts almost transparent, like the NeRF model, so the first proposals are uniform
        self.density_layer[0].bias.data[0] = -1.5

    def _ray_points(self, ray_bundle):
        """Same sample points as the NeRF model, contracted if enabled."""
        points = ray_bundle_to_ray_points(ray_bundle)
        if self.contract:
            points = contract(points, self.contraction_center, self.contraction_radius)
        return points

    def forward(self, ray_bundle: RayBundle, **kwargs):
        """Returns the opacity (..., P, 1) of every sample point along the rays."""
        points = self._ray_points(ray_bundle)
        if self.architecture == "hashgrid":
            x, inside = normalize_points(points, self.contract, self.contraction_center,
                                         self.contraction_radius, self.scene_bound)
            raw_densities = trunc_exp(self.mlp(self.encoding(x)) - 1.0) * inside
        else:
            raw_densities = self.density_layer(self.mlp(self.harmonic_embedding(points)))
        return density_to_alpha(raw_densities, ray_bundle.lengths, ray_bundle.directions, uses_delta=True)

    def batched_forward(self, ray_bundle: RayBundle, n_batches: int = N_BATCHES_FULL_RENDER, **kwargs):
        """Runs forward in chunks so a full image can be rendered without running out of memory."""
        return chunked_forward(lambda bundle, **_: (self.forward(bundle),), ray_bundle, n_batches)[0]