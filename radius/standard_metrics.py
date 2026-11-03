# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Standard distributional alignment metrics from prior work, used for comparison with RADIUS."""
from typing import List
import numpy as np
from scipy.stats import chi2_contingency


def _normalize(dist: List[int]) -> np.ndarray:
    total = sum(dist)
    return np.array(dist) / total if total > 0 else np.zeros(len(dist))


def cramers_v(human_dist: List[int], simulated_dist: List[int]) -> float:
    """Cramér's V — association strength between two distributions (0-1)."""
    data = np.array([human_dist, simulated_dist]) + 1e-10
    try:
        chi2, _, _, _ = chi2_contingency(data)
        n = np.sum(data)
        min_dim = min(data.shape) - 1
        return np.sqrt(chi2 / (n * min_dim))
    except ValueError:
        return 0.0


def js_divergence(human_dist: List[int], simulated_dist: List[int]) -> float:
    """Jensen-Shannon divergence (0-1)."""
    epsilon = 1e-10
    p = _normalize(human_dist) + epsilon
    q = _normalize(simulated_dist) + epsilon
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def wasserstein_distance(human_dist: List[int], simulated_dist: List[int]) -> float:
    """Wasserstein (earth mover's) distance between distributions."""
    p = _normalize(human_dist)
    q = _normalize(simulated_dist)
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))))


def simbench_score(human_dist: List[int], simulated_dist: List[int]) -> float:
    """SimBench score (Hu et al., 2025): S = 100 * (1 - TVD(P,Q) / TVD(P,U)).

    Measures improvement over uniform baseline. Range: (-inf, 100].
    100 = perfect, 0 = no better than uniform, <0 = worse than uniform.
    """
    n = len(human_dist)
    p = _normalize(human_dist)
    q = _normalize(simulated_dist)
    u = np.ones(n) / n
    tvd_pq = 0.5 * np.sum(np.abs(p - q))
    tvd_pu = 0.5 * np.sum(np.abs(p - u))
    if tvd_pu == 0:
        return 100.0 if tvd_pq == 0 else 0.0
    return float(100.0 * (1.0 - tvd_pq / tvd_pu))
