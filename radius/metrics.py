# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from scipy.stats import chi2_contingency

def _normalize_distribution(dist: List[int]) -> np.ndarray:
    """Normalize distribution to probabilities."""
    total = sum(dist)
    return np.array(dist) / total if total > 0 else np.zeros(len(dist))


def total_variance_distance(human_dist: List[int], simulated_dist: List[int]) -> float:
    """Calculate Total Variation Distance between distributions.

    Args:
        human_dist: Human response counts
        simulated_dist: Simulated response counts

    Returns:
        Total variation distance (0-1)
    """
    human_probs = _normalize_distribution(human_dist)
    sim_probs = _normalize_distribution(simulated_dist)

    return np.sum(np.abs(human_probs - sim_probs)) / 2


def top_n_rank_correlation(human_dist: List[int], simulated_dist: List[int]) -> float:
    """Calculate rank correlation, optionally focusing on top N choices.
    
    Args:
        human_dist: Human response counts
        simulated_dist: Simulated response counts
        
    Returns:
        Normalized rank correlation (0-1)
    """
    human_probs = _normalize_distribution(human_dist)
    sim_probs = _normalize_distribution(simulated_dist)

    if np.allclose(human_probs, sim_probs):
        return 1.0

    # Handle ties by using unique values and their inverse mapping
    unique_human, human_inverse = np.unique(human_probs, return_inverse=True)
    unique_sim, sim_inverse = np.unique(sim_probs, return_inverse=True)

    # Get proper ranks for unique values (higher probability = lower rank number)
    human_unique_ranks = np.empty_like(unique_human, dtype=int)
    human_unique_ranks[np.argsort(-unique_human)] = np.arange(len(unique_human))
    sim_unique_ranks = np.empty_like(unique_sim, dtype=int)
    sim_unique_ranks[np.argsort(-unique_sim)] = np.arange(len(unique_sim))

    # Map back to original arrays
    human_ranks = human_unique_ranks[human_inverse]
    sim_ranks = sim_unique_ranks[sim_inverse]

    if len(np.unique(human_ranks)) < 2 or len(np.unique(sim_ranks)) < 2:
        return 0.0

    mean_human = np.mean(human_ranks)
    mean_sim = np.mean(sim_ranks)
    num = np.sum((human_ranks - mean_human) * (sim_ranks - mean_sim))
    denom = np.sqrt(np.sum((human_ranks - mean_human) ** 2) * np.sum((sim_ranks - mean_sim) ** 2))
    rank_corr = num / denom if denom != 0 else 0.0

    if np.isnan(rank_corr):
        return 0.0

    return (rank_corr + 1) / 2


def _bootstrap_rank_significance(
    human_counts: List[int], n_bootstrap: int = 1000, confidence_level: float = 0.95
) -> Dict[str, Any]:
    """Bootstrap human votes to determine significance groups for ranking.

    Args:
        human_counts: Vote counts for each option
        n_bootstrap: Number of bootstrap samples. Defaults to 1000
        confidence_level: Confidence level for significance testing. Defaults to 0.95

    Returns:
        Dictionary containing significance groups and pairwise comparisons
    """
    n_options = len(human_counts)
    total_votes = sum(human_counts)

    # Create individual vote array
    votes = []
    for option_idx, count in enumerate(human_counts):
        votes.extend([option_idx] * count)

    # Bootstrap sampling
    bootstrap_percentages = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(votes, size=total_votes, replace=True)
        bootstrap_counts = [np.sum(bootstrap_sample == i) for i in range(n_options)]
        bootstrap_pct = [count / total_votes * 100 for count in bootstrap_counts]
        bootstrap_percentages.append(bootstrap_pct)

    bootstrap_percentages = np.array(bootstrap_percentages)

    # Calculate confidence intervals for each option
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_percentages, (alpha / 2) * 100, axis=0)
    ci_upper = np.percentile(bootstrap_percentages, (1 - alpha / 2) * 100, axis=0)

    # Determine significance groups by checking CI overlaps
    # Sort options by their mean percentage (descending)
    mean_percentages = np.mean(bootstrap_percentages, axis=0)
    sorted_indices = np.argsort(mean_percentages)[::-1]

    significance_groups = []
    current_group = [sorted_indices[0]]

    for i in range(1, len(sorted_indices)):
        current_idx = sorted_indices[i]
        top_idx = sorted_indices[0]

        # Check if current option's CI overlaps with top option's CI
        overlap = not (
            ci_upper[current_idx] < ci_lower[top_idx]
            or ci_lower[current_idx] > ci_upper[top_idx]
        )

        if overlap:
            current_group.append(current_idx)
        else:
            significance_groups.append(current_group)
            current_group = [current_idx]

    significance_groups.append(current_group)

    return {
        "significance_groups": significance_groups,
        "top_group": significance_groups[0],
        "mean_percentages": mean_percentages,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def top_rank_match(human_dist: List[int], simulated_dist: List[int]) -> int:
    """Check if agent's top-ranked choice matches human's statistically significant top group.

    Uses bootstrap significance testing to determine which options are statistically
    tied for top rank in human responses, then checks if the agent's highest-scoring
    option falls within this significance group.

    Args:
        human_dist: Human response counts for each option
        simulated_dist: Agent/simulated response counts for each option

    Returns:
        1 if agent's top choice is in human's significant top group, 0 otherwise
    """
    human_sig = _bootstrap_rank_significance(human_dist)
    agent_top1 = list(np.where(simulated_dist == np.max(simulated_dist))[0])
    human_top1 = human_sig["top_group"]

    agent_top1.sort()
    human_top1.sort()

    # Penalize ties in agents top rank when there is clear preference in human score
    # Example: H:[50, 30, 15, 5], A:[30, 30, 25, 15] --> top rank match = 0
    if len(agent_top1) > 1:
        return int(agent_top1 == human_top1)
    else:
        agent_top1 = agent_top1[0]

    # If human top-1 is in a significance group (tied), agent is correct if it picks any from that group
    return int(agent_top1 in human_top1)


def chi_square_distributions_homogeneity(human_dist: List[int], simulated_dist: List[int]) -> float:
    """Calculate p-value from Chi-Square Test for Homogeneity using scipy.
    
    Tests whether two response distributions are statistically significantly different.
    
    Args:
        human_dist: Human response distribution counts
        simulated_dist: Simulated response distribution counts
        
    Returns:
        P-value (P<0.05 indicates statistically significant difference)
    """    
    data = np.array([human_dist, simulated_dist]) + 1e-10
    try:
        _, p_value, _, _ = chi2_contingency(data)
        distribution_homogeneity = float(p_value >= 0.05)
        return distribution_homogeneity
    except (ValueError, ZeroDivisionError) as e:
        import warnings
        warnings.warn(f"Chi-square test failed: {e}. Returning 0.0. Distributions: human={human_dist}, simulated={simulated_dist}")
        return 0.0
