import torch
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points

from src.repo_util.LoadConfigurations import (
    N_HARMONIC_FUNCTIONS,
    HARMONIC_OMEGA0,
    N_HIDDEN_NEURONS,
    N_BATCHES_FULL_RENDER,
    DENSITY_USES_DELTA,
)

"""
Baseline model from the course book (Chapter 6), with one change for the final project:
the opacity can take the spacing between samples into account (NeRF, Mildenhall et al. 2020, eq. 3).
That is needed for hierarchical sampling, where the samples along a ray are no longer evenly spaced.
Set sampler.density_uses_delta: false in config.yaml to get the original book behaviour.
"""


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=N_HARMONIC_FUNCTIONS, omega0=HARMONIC_OMEGA0):
        """Turns each input coordinate into sine and cosine values at increasing frequencies."""
        super().__init__()
        # Frequencies are omega0 times 1, 2, 4, 8 and so on
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x):
        """Returns the embedding of x, with n_harmonic_functions * 2 values per coordinate."""
        # Multiplies every coordinate by every frequency, then flattens them into one vector
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions=N_HARMONIC_FUNCTIONS,
        n_hidden_neurons=N_HIDDEN_NEURONS,
        omega0=HARMONIC_OMEGA0,
        density_uses_delta=DENSITY_USES_DELTA,
    ):
        """Builds the MLP that predicts opacity and colour for each sample point."""
        super().__init__()
        self.density_uses_delta = density_uses_delta

        # Used for both the point positions and the viewing directions
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions, omega0)

        # Sine and cosine for each frequency and each of the 3 coordinates
        embedding_dim = n_harmonic_functions * 2 * 3

        # Shared part with two hidden layers, Softplus is used instead of ReLU
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
        )

        # Colour head, takes the shared features together with the embedded viewing direction
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons + embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, 3),
            torch.nn.Sigmoid(),
            # Sigmoid keeps the RGB values between 0 and 1
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

    def _get_densities(self, features, lengths, directions):
        """Predicts the opacity of each sample point, mapped to a value between 0 and 1.

        features:   (..., P, n_hidden)  per-point features
        lengths:    (..., P)            distance of each sample along its ray
        directions: (..., 3)            ray directions (not unit length in PyTorch3D)
        """
        raw_densities = self.density_layer(features)  # (..., P, 1)

        if not self.density_uses_delta:
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

    def _get_colors(self, features, rays_directions):
        """Predicts the RGB colour of each point from its features and the viewing direction."""
        spatial_size = features.shape[:-1]

        # Scales the directions to unit length
        rays_directions_normed = torch.nn.functional.normalize(
            rays_directions, dim=-1
        )

        rays_embedding = self.harmonic_embedding(
            rays_directions_normed
        )

        # Every point on a ray shares the same direction, so the embedding is repeated per point
        rays_embedding_expand = rays_embedding[..., None, :].expand(
            *spatial_size, rays_embedding.shape[-1]
        )

        color_layer_input = torch.cat(
            (features, rays_embedding_expand),
            dim=-1
        )
        return self.color_layer(color_layer_input)

    def forward(
        self,
        ray_bundle: RayBundle,
        **kwargs,
    ):
        """Returns the opacity and RGB colour of every sample point along the rays."""
        # Turns ray origins, directions and lengths into 3D sample points
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]

        embeds = self.harmonic_embedding(
            rays_points_world
        )
        # embeds.shape = [minibatch x ... x self.n_harmonic_functions*6]

        # Shared features for both heads
        features = self.mlp(embeds)
        # features.shape = [minibatch x ... x n_hidden_neurons]

        rays_densities = self._get_densities(features, ray_bundle.lengths, ray_bundle.directions)
        # rays_densities.shape = [minibatch x ... x 1]

        rays_colors = self._get_colors(features, ray_bundle.directions)
        # rays_colors.shape = [minibatch x ... x 3]

        return rays_densities, rays_colors

    def batched_forward(
        self,
        ray_bundle: RayBundle,
        n_batches: int = N_BATCHES_FULL_RENDER,
        **kwargs,
    ):
        """Runs forward in chunks so a full image can be rendered without running out of memory."""

        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        # Splits all rays into n_batches chunks
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        # reshape instead of view, since lengths from the fine pass may not be contiguous
        origins = ray_bundle.origins.reshape(-1, 3)
        directions = ray_bundle.directions.reshape(-1, 3)
        lengths = ray_bundle.lengths.reshape(-1, n_pts_per_ray)

        # Runs the normal forward pass on one chunk at a time
        batch_outputs = [
            self.forward(
                RayBundle(
                    origins=origins[batch_idx],
                    directions=directions[batch_idx],
                    lengths=lengths[batch_idx],
                    xys=None,
                )
            ) for batch_idx in batches
        ]

        # Joins the chunks again and restores the original shape
        rays_densities, rays_colors = [
            torch.cat(
                [batch_output[output_i] for batch_output in batch_outputs], dim=0
            ).view(*spatial_size, -1) for output_i in (0, 1)
        ]
        return rays_densities, rays_colors