# RADIUS: Alignment Suite for Survey Simulation

This package contains the alignment metrics proposed in the _"RADIUS: Ranking, Distribution, and Significance — A Comprehensive Alignment Suite for Survey Simulation"_ paper for evaluating survey simulation quality. We propose an evaluation approach for survey simulation that extends beyond simple "winner" option identification to assess whether simulated surveys capture the underlying patterns and variability present in human responses. 

## Metrics

We define alignment through two complementary criteria:

- **Ranking alignment**: measuring how well the relative ordering of selected options is preserved and verifying the alignment in the top option selection.
- **Distribution alignment**: quantifying how much probability "mass" must be redistributed to transform agent distribution into the target human one and testing the presence of statistically significant differences between response distributions.

### Ranking Alignment

#### Top Rank Match
Function: `top_rank_match(human_dist, simulated_dist) -> int`

Checks if the agent's top-ranked choice matches the human's statistically significant top group using bootstrap significance testing (1000 samples, 95% confidence). Returns 1 if the agent's top choice falls within the human's significant top group, 0 otherwise. Penalizes agent ties when humans show clear preference.

Range: {0, 1}  
Interpretation: 1 = perfect top-rank alignment, 0 = misalignment

#### Rank Correlation
Function: `top_n_rank_correlation(human_dist, simulated_dist) -> float`

Calculates Spearman correlation between option ranks in human and simulated distributions, normalized to [0, 1]. Handles ties by assigning unique ranks to probability values. Higher probability receives lower rank number (rank 0 = most preferred).

Range: [0, 1]  
Interpretation: 1 = perfect rank correlation, 0 = no correlation

### Distribution Alignment

#### Total Variation Distance
Function: `total_variance_distance(human_dist, simulated_dist) -> float`

Measures the minimum probability mass that must be redistributed to transform one distribution into another. Computed as half the L1 distance between normalized probability distributions.

Range: [0, 1]  
Interpretation: 0 = identical distributions, 1 = completely misaligned distributions

#### Chi-Square Test for Homogeneity
Function: `chi_square_distributions_homogeneity(human_dist, simulated_dist) -> float`

Tests whether two response distributions are statistically significantly different using the chi-square test. Returns 1 if distributions are homogeneous (p ≥ 0.05), 0 if significantly different (p < 0.05).

Range: {0, 1}  
Interpretation: 1 = no statistical difference between distributions, 0 = distributions differ significantly

## Alignment Suite Execution

### Question-Level Analysis

For each survey question, compute all four metrics:

```python
from radius import (
    top_rank_match,
    top_n_rank_correlation,
    total_variance_distance,
    chi_square_distributions_homogeneity
)

# Example: Human responses [50, 30, 15, 5], Agent responses [45, 35, 15, 5]
human_dist = [50, 30, 15, 5]
simulated_dist = [45, 35, 15, 5]

metrics = {
    'top_rank_match': top_rank_match(human_dist, simulated_dist),
    'rank_correlation': top_n_rank_correlation(human_dist, simulated_dist),
    'total_variation_distance': total_variance_distance(human_dist, simulated_dist),
    'chi_square_homogeneity': chi_square_distributions_homogeneity(human_dist, simulated_dist)
}
```

### Survey-Level Aggregation

Aggregate question-level metrics across all questions in a survey:

```python
import numpy as np

# Collect metrics for all questions
question_metrics = []
for question_id, (human_dist, sim_dist) in questions.items():
    metrics = compute_question_metrics(human_dist, sim_dist)
    question_metrics.append(metrics)

# Calculate survey-level averages
survey_alignment = {
    'avg_top_rank_match': np.mean([m['top_rank_match'] for m in question_metrics]),
    'avg_rank_correlation': np.mean([m['rank_correlation'] for m in question_metrics]),
    'avg_total_variation_distance': np.mean([m['total_variation_distance'] for m in question_metrics]),
    'avg_chi_square_homogeneity': np.mean([m['chi_square_homogeneity'] for m in question_metrics])
}
```

### Statistical Significance Testing

Compare simulation performance against baselines using paired t-tests:

```python
from scipy import stats

# Collect per-question metrics for simulation and baseline
sim_scores = [metric_value for question in questions]
baseline_scores = [metric_value for question in questions]

# Paired t-test (questions are paired)
t_stat, p_value = stats.ttest_rel(sim_scores, baseline_scores)

# Mark significant improvements (p < 0.05)
is_significant = p_value < 0.05
```
