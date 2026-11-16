import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from stability.hyperbolic_kmeans.hkmeans import HyperbolicKMeans
from stability.hyperbolic_kmeans.util_hk import poincare_dist
from stability.hypdiff_models import StabilityDiffusion, StabilityLPModel
from stability.edge_perturbation import remove_edges
from stability.partial_recovery import (
    compute_community_context,
    decode_without_diffusion,
    decode_edge_set,
    reconstruct_k,
    score_reconstructions,
    validate_curvature,
)
from stability.pilot_config import PilotConfig


def make_args(curvature=1.0):
    return SimpleNamespace(
        manifold="PoincareBall",
        c=curvature,
        cuda=-1,
        device="cpu",
        feat_dim=3,
        dim=4,
        model="HGCN",
        n_nodes=4,
        num_layers=2,
        act="relu",
        dropout=0.0,
        bias=0,
        use_att=0,
        local_agg=0,
        r=2.0,
        t=1.0,
        nb_false_edges=1,
        nb_edges=2,
        hid1=8,
        hid2=4,
    )


def make_adjacency(node_count, edges):
    rows = list(range(node_count))
    cols = list(range(node_count))
    for source, target in edges:
        rows.extend((source, target))
        cols.extend((target, source))
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(
        indices, values, (node_count, node_count)
    ).coalesce()
    degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp_min(1)
    normalized_values = adjacency.values() / torch.sqrt(
        degree[adjacency.indices()[0]] * degree[adjacency.indices()[1]]
    )
    return torch.sparse_coo_tensor(
        adjacency.indices(), normalized_values, adjacency.shape
    ).coalesce()


class PartialRecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1432)
        np.random.seed(19)
        self.args = make_args()
        self.config = PilotConfig(curvature=1.0, tau=0.5)
        self.config.apply_to_args(self.args)
        self.encoder = StabilityLPModel(self.args).cpu().eval()
        self.denoiser = StabilityDiffusion(
            self.args, dim=self.args.dim, curvature=self.config.curvature
        ).cpu().eval()
        self.clusterer = HyperbolicKMeans(n_clusters=2, curvature=1.0)
        self.node_ids = ("a", "b", "c", "d")
        self.features = torch.tensor(
            [
                [0.1, 0.2, 0.0],
                [0.2, -0.1, 0.1],
                [-0.1, 0.1, 0.2],
                [0.0, -0.2, 0.1],
            ]
        )
        self.original_edges = (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"))
        self.original_adjacency = make_adjacency(4, ((0, 1), (1, 2), (2, 3), (3, 0)))

    def test_curvature_is_explicit_and_clustering_distance_matches_model(self):
        validate_curvature(self.encoder, self.denoiser, self.clusterer, self.config)
        self.assertEqual(self.denoiser.c, 1.0)
        self.assertTrue(
            all(
                np.isclose(float(curvature.detach().reshape(-1)[0]), 1.0)
                for curvature in self.encoder.encoder.curvatures
            )
        )
        with self.assertRaisesRegex(ValueError, "curvature=1.0"):
            HyperbolicKMeans(n_clusters=2, curvature=0.6)

        point1 = torch.tensor([0.1, 0.2, -0.1], dtype=torch.float64)
        point2 = torch.tensor([-0.2, 0.05, 0.15], dtype=torch.float64)
        model_distance = torch.sqrt(
            self.encoder.manifold.sqdist(
                point1.unsqueeze(0), point2.unsqueeze(0), self.encoder.c
            )
        ).item()
        cluster_distance = poincare_dist(
            point1.numpy(), point2.numpy(), c=self.clusterer.curvature
        )
        self.assertAlmostEqual(model_distance, cluster_distance, places=10)
        embeddings = torch.stack((point1.float(), point2.float()))
        edge_index = torch.tensor([[0, 1]], dtype=torch.long)
        actual_probability = self.encoder.decode(embeddings, edge_index)
        expected_probability = self.encoder.dc(
            self.encoder.manifold.sqdist(
                embeddings[0:1], embeddings[1:2], self.encoder.c
            )
        )
        torch.testing.assert_close(actual_probability, expected_probability)

    def test_original_context_is_finite_and_inside_unit_ball(self):
        context = compute_community_context(
            self.encoder,
            self.features,
            self.original_adjacency,
            self.clusterer,
            self.node_ids,
        )
        with torch.no_grad():
            embeddings = self.encoder.encode(
                self.features, self.original_adjacency
            )[2]
        self.assertTrue(torch.isfinite(embeddings).all())
        self.assertTrue(torch.isfinite(context.centers).all())
        self.assertTrue(torch.all(torch.linalg.vector_norm(embeddings, dim=1) < 1))
        self.assertTrue(torch.all(torch.linalg.vector_norm(context.centers, dim=1) < 1))
        self.assertEqual(context.labels.shape, (4,))

    def test_context_projects_boundary_centers_before_recovery_logmap(self):
        from manifolds.poincare import PoincareBall

        embeddings = torch.tensor([[0.97, 0.0], [0.96, 0.01]], dtype=torch.float32)

        class Encoder:
            def encode(self, features, adjacency):
                return embeddings, None, embeddings, adjacency

        class Clusterer:
            curvature = 1.0
            manifold = PoincareBall()

            def fit(self, _points):
                return np.asarray([[1.0, 0.0]]), np.asarray([0, 0])

        context = compute_community_context(
            Encoder(),
            features=embeddings,
            original_adjacency=torch.empty((2, 2)),
            clusterer=Clusterer(),
            node_ids=("a", "b"),
        )
        self.assertTrue(torch.isfinite(context.centers).all())
        self.assertTrue(
            torch.all(torch.linalg.vector_norm(context.centers, dim=1) < 1.0)
        )
        tangent = PoincareBall().logmap(
            context.centers[context.labels], embeddings, c=1.0
        )
        self.assertTrue(torch.isfinite(tangent).all())

    def test_noise_formula_round_trip_at_index_zero_and_nonzero(self):
        x_start = torch.tensor(
            [[0.2, -0.3, 0.4, -0.1], [-0.4, 0.1, 0.3, 0.2]]
        )
        noise = torch.tensor(
            [[0.5, 0.25, -0.75, 0.125], [-0.5, -0.25, 0.75, -0.125]]
        )
        for timestep in (0, 7):
            times = torch.full((2,), timestep, dtype=torch.long)
            x_t = self.denoiser.q_sample(
                x_start,
                times,
                direction=None,
                noise=noise,
                labels=None,
                u0=None,
                restrict=False,
            )
            recovered_noise = self.denoiser.predict_noise_from_start(
                x_t, times, x_start
            )
            if timestep == 0:
                self.assertFalse(torch.allclose(x_t, x_start.to(x_t.dtype)))
            torch.testing.assert_close(
                recovered_noise, noise.to(recovered_noise.dtype), rtol=1e-6, atol=1e-7
            )

    def test_tstar_zero_denoises_once_and_returns_clean_prediction(self):
        x_start = torch.full((4, 4), 0.2)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        direction = torch.ones((1, 4))
        labels = torch.zeros(4, dtype=torch.long)
        expected_clean = torch.full((4, 4), 0.3)
        captured_times = []

        class FixedDenoiser(torch.nn.Module):
            def forward(self, _x, _adj, time):
                captured_times.append(time.clone())
                return expected_clean

        with patch.object(self.denoiser, "gcn", FixedDenoiser()):
            actual_clean = self.denoiser._denoise_from_tstar(
                x_start,
                edge_index,
                direction,
                labels,
                0,
                torch.Generator().manual_seed(7),
            )

        torch.testing.assert_close(actual_clean, expected_clean)
        self.assertEqual(len(captured_times), 1)
        torch.testing.assert_close(captured_times[0], torch.zeros(4))
        with self.assertRaisesRegex(ValueError, "t_star"):
            self.denoiser._denoise_from_tstar(
                x_start,
                edge_index,
                direction,
                labels,
                self.denoiser.num_timesteps,
                torch.Generator().manual_seed(7),
            )

    def test_reverse_loop_uses_normalized_time_and_visits_each_index(self):
        x_start = torch.full((4, 4), 0.2)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        direction = torch.ones((1, 4))
        labels = torch.zeros(4, dtype=torch.long)
        captured_times = []

        hook = self.denoiser.gcn.register_forward_pre_hook(
            lambda _module, inputs: captured_times.append(inputs[2].clone())
        )
        try:
            self.denoiser._denoise_from_tstar(
                x_start,
                edge_index,
                direction,
                labels,
                2,
                torch.Generator().manual_seed(5),
            )
        finally:
            hook.remove()

        self.assertEqual(len(captured_times), 3)
        for actual, expected in zip(captured_times, (0.01, 0.005, 0.0)):
            torch.testing.assert_close(actual, torch.full((4,), expected))
    def test_partial_reconstruction_uses_damaged_graph_and_independent_streams(self):
        node_index = {node: index for index, node in enumerate(self.node_ids)}
        perturbed = remove_edges(
            self.node_ids, self.features, self.original_edges, 1, seed=4
        )
        removed = set(perturbed.removed_edges)
        self.assertEqual(len(removed), 1)
        damaged_edges = tuple(
            (node_index[source], node_index[target]) for source, target in perturbed.edges
        )
        damaged_adjacency = make_adjacency(4, damaged_edges)
        observed = set(
            zip(
                damaged_adjacency.indices()[0].tolist(),
                damaged_adjacency.indices()[1].tolist(),
            )
        )
        self.assertTrue(
            all(
                (node_index[source], node_index[target]) not in observed
                and (node_index[target], node_index[source]) not in observed
                for source, target in removed
            )
        )

        context = compute_community_context(
            self.encoder,
            self.features,
            self.original_adjacency,
            self.clusterer,
            self.node_ids,
        )
        frozen_labels = context.labels.clone()
        frozen_centers = context.centers.clone()
        with torch.no_grad():
            damaged_embeddings = self.encoder.encode(
                self.features, damaged_adjacency
            )[2]
            self.assertTrue(torch.isfinite(damaged_embeddings).all())
            self.assertTrue(
                torch.all(torch.linalg.vector_norm(damaged_embeddings, dim=1) < 1)
            )
            recovered_embeddings = self.denoiser.sample_from_tstar(
                damaged_embeddings,
                damaged_adjacency,
                context.labels,
                context.centers,
                t_star=2,
                generator=torch.Generator().manual_seed(22),
            )
        self.assertEqual(recovered_embeddings.shape, (4, self.args.dim))
        self.assertTrue(torch.isfinite(recovered_embeddings).all())
        self.assertTrue(
            torch.all(torch.linalg.vector_norm(recovered_embeddings, dim=1) < 1)
        )

        signature = inspect.signature(self.denoiser.sample_from_tstar)
        self.assertNotIn("original_adjacency", signature.parameters)
        self.assertNotIn("removed_edges", signature.parameters)
        recovery_signature = inspect.signature(reconstruct_k)
        self.assertNotIn("original_adjacency", recovery_signature.parameters)
        self.assertNotIn("removed_edges", recovery_signature.parameters)

        original_draw_noise = self.denoiser.draw_noise
        captured_noise = []

        def record_noise(x_start, generator):
            noise = original_draw_noise(x_start, generator)
            captured_noise.append(noise.clone())
            return noise

        candidates = tuple(
            (source, target)
            for index, source in enumerate(self.node_ids)
            for target in self.node_ids[index + 1 :]
        )
        with patch.object(self.denoiser, "draw_noise", side_effect=record_noise):
            first_batch = reconstruct_k(
                self.encoder,
                self.denoiser,
                damaged_embeddings,
                damaged_adjacency,
                context,
                self.node_ids,
                candidates,
                t_star=2,
                num_reconstructions=4,
                seed=23,
                config=self.config,
            )
        first_noise = [noise.clone() for noise in captured_noise]
        captured_noise.clear()
        with patch.object(self.denoiser, "draw_noise", side_effect=record_noise):
            second_batch = reconstruct_k(
                self.encoder,
                self.denoiser,
                damaged_embeddings,
                damaged_adjacency,
                context,
                self.node_ids,
                candidates,
                t_star=2,
                num_reconstructions=4,
                seed=23,
                config=self.config,
            )

        self.assertEqual(len(first_noise), 4)
        self.assertTrue(
            all(
                not torch.equal(first_noise[i], first_noise[j])
                for i in range(4)
                for j in range(i + 1, 4)
            )
        )
        for first, second in zip(first_noise, captured_noise):
            torch.testing.assert_close(first, second)
        self.assertEqual(first_batch.edge_sets, second_batch.edge_sets)
        self.assertTrue(torch.equal(context.labels, frozen_labels))
        self.assertTrue(torch.equal(context.centers, frozen_centers))
        self.assertEqual(len(first_batch.edge_sets), 4)
        self.assertTrue(
            all(
                torch.isfinite(probabilities).all()
                and torch.all((probabilities >= 0) & (probabilities <= 1))
                for probabilities in first_batch.probabilities
            )
        )
        score = score_reconstructions(first_batch, removed)
        self.assertTrue(0.0 <= score.scenario_score <= 1.0)

    def test_no_diffusion_baseline_decodes_damaged_embeddings_directly(self):
        damaged_adjacency = make_adjacency(4, ((0, 1), (1, 2), (2, 3)))
        with torch.no_grad():
            damaged_embeddings = self.encoder.encode(
                self.features, damaged_adjacency
            )[2]
        candidates = (("a", "b"), ("a", "c"))
        edges, probabilities = decode_without_diffusion(
            self.encoder,
            damaged_embeddings,
            self.node_ids,
            candidates,
            tau=0.5,
        )
        self.assertTrue(set(edges).issubset(candidates))
        self.assertEqual(probabilities.shape, (2,))

    def test_tau_is_explicit_and_inclusive_at_the_boundary(self):
        class FixedDecoder:
            def decode(self, _embeddings, _edge_index):
                return torch.tensor([0.5, 0.499])

        edge_set, probabilities = decode_edge_set(
            FixedDecoder(),
            torch.zeros((3, 2)),
            ("a", "b", "c"),
            (("a", "b"), ("a", "c")),
            tau=0.5,
        )
        self.assertEqual(edge_set, (("a", "b"),))
        torch.testing.assert_close(probabilities, torch.tensor([0.5, 0.499]))


if __name__ == "__main__":
    unittest.main()
