from __future__ import annotations

import math

import networkx as nx
import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist


def graph_feature_histograms(node_ids, edges):
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(tuple(edge) for edge in edges)
    if len(graph) != len(node_ids) or set(graph.nodes) != set(node_ids):
        raise ValueError("graph feature extraction changed the source node set")
    if nx.number_of_selfloops(graph):
        raise ValueError("self-loops are not supported")

    degree = np.asarray(nx.degree_histogram(graph), dtype=np.float64)
    clustering = np.asarray(
        list(nx.clustering(graph).values()), dtype=np.float64
    )
    if len(clustering) != len(node_ids) or not np.isfinite(clustering).all():
        raise ValueError("invalid clustering coefficients")
    clustering = np.histogram(
        clustering, bins=100, range=(0.0, 1.0)
    )[0].astype(np.float64)

    if graph.number_of_nodes() == 0:
        raise ValueError("zero-node graphs are not supported")
    laplacian = nx.normalized_laplacian_matrix(graph).toarray()
    eigenvalues = scipy.linalg.eigvalsh(laplacian, check_finite=True)
    tolerance = 1e-12
    if (
        len(eigenvalues) != len(node_ids)
        or not np.isfinite(eigenvalues).all()
        or np.any(eigenvalues < -1e-5 - tolerance)
        or np.any(eigenvalues > 2.0 + tolerance)
    ):
        raise ValueError("invalid normalized-Laplacian eigenvalues")
    eigenvalues = np.clip(eigenvalues, -1e-5, 2.0)
    spectral = np.histogram(
        eigenvalues, bins=200, range=(-1e-5, 2.0)
    )[0].astype(np.float64)
    if int(spectral.sum()) != len(node_ids):
        raise ValueError("spectral histogram dropped eigenvalues")

    features = []
    for histogram in (degree, clustering, spectral):
        total = float(histogram.sum())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("invalid graph histogram")
        features.append(histogram / total)
    return tuple(features)


def feature_distance_matrix(features):
    if not features:
        raise ValueError("at least one graph feature vector is required")
    degree_size = max(len(row[0]) for row in features)
    degree = np.zeros((len(features), degree_size), dtype=np.float64)
    clustering = np.empty((len(features), 100), dtype=np.float64)
    spectral = np.empty((len(features), 200), dtype=np.float64)
    for index, row in enumerate(features):
        if len(row) != 3:
            raise ValueError("expected degree, clustering, and spectral histograms")
        if len(row[1]) != 100 or len(row[2]) != 200:
            raise ValueError("unexpected clustering or spectral histogram size")
        degree[index, : len(row[0])] = row[0]
        clustering[index] = row[1]
        spectral[index] = row[2]
    distance = (
        cdist(degree, degree, metric="cityblock")
        + cdist(clustering, clustering, metric="cityblock")
        + cdist(spectral, spectral, metric="cityblock")
    ) / 6.0
    if not np.isfinite(distance).all():
        raise ValueError("non-finite graph feature distance")
    np.fill_diagonal(distance, 0.0)
    return distance


def prdc_from_distances(real_real, generated_generated, real_generated, k_values):
    real_real = np.asarray(real_real, dtype=np.float64)
    generated_generated = np.asarray(generated_generated, dtype=np.float64)
    real_generated = np.asarray(real_generated, dtype=np.float64)
    real_count, generated_count = real_generated.shape
    if real_real.shape != (real_count, real_count):
        raise ValueError("real distance matrix has the wrong shape")
    if generated_generated.shape != (generated_count, generated_count):
        raise ValueError("generated distance matrix has the wrong shape")
    if not (
        np.isfinite(real_real).all()
        and np.isfinite(generated_generated).all()
        and np.isfinite(real_generated).all()
    ):
        raise ValueError("distance matrices must be finite")
    if real_count < 2 or generated_count < 2:
        raise ValueError("at least two real and generated samples are required")

    real_neighbors = real_real.copy()
    generated_neighbors = generated_generated.copy()
    np.fill_diagonal(real_neighbors, np.inf)
    np.fill_diagonal(generated_neighbors, np.inf)
    real_sorted = np.sort(real_neighbors, axis=1)
    generated_sorted = np.sort(generated_neighbors, axis=1)
    results = {}
    for k in k_values:
        if not 1 <= k < min(real_count, generated_count):
            raise ValueError("k must be positive and smaller than both sample counts")
        real_radii = real_sorted[:, k - 1]
        generated_radii = generated_sorted[:, k - 1]

        in_real_neighborhoods = real_generated <= real_radii[:, None]
        in_generated_neighborhoods = (
            real_generated <= generated_radii[None, :]
        )
        precision = float(
            np.mean(np.any(in_real_neighborhoods, axis=0))
        )
        recall = float(
            np.mean(np.any(in_generated_neighborhoods, axis=1))
        )
        density = float(
            np.count_nonzero(in_real_neighborhoods) / (k * generated_count)
        )
        coverage = float(
            np.mean(np.any(in_real_neighborhoods, axis=1))
        )
        f1pr = (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
        f1dc = (
            0.0
            if density + coverage == 0.0
            else 2.0 * density * coverage / (density + coverage)
        )
        results[int(k)] = {
            "precision": precision,
            "recall": recall,
            "density": density,
            "coverage": coverage,
            "f1pr": f1pr,
            "f1dc": f1dc,
        }
    return results
