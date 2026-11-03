# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Evaluation for survey simulation — wraps RADIUS alignment suite."""

from __future__ import annotations
import json, re
from typing import Dict, List

import numpy as np

from radius import (
    total_variance_distance,
    top_n_rank_correlation,
    top_rank_match,
    chi_square_distributions_homogeneity,
)
from radius.standard_metrics import cramers_v, js_divergence, wasserstein_distance, simbench_score

# ---------------------------------------------------------------------------
# All-in-one evaluator using RADIUS metrics
# ---------------------------------------------------------------------------

METRICS = {
    "TRM": top_rank_match,
    "RC": top_n_rank_correlation,
    "TVD": total_variance_distance,
    "DH": chi_square_distributions_homogeneity,
    "CV": cramers_v,
    "JSD": js_divergence,
    "WD": wasserstein_distance,
    "SS": simbench_score,
}


def compute_alignment(human_counts: List[int], agent_counts: List[int]) -> Dict[str, float]:
    """Compute all RADIUS alignment metrics for a single question."""
    return {name: fn(human_counts, agent_counts) for name, fn in METRICS.items()}


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------

def parse_choice(llm_output: str, valid_options: List[str]) -> str | None:
    """Extract a single selected answer from LLM output."""
    if not llm_output or not llm_output.strip():
        return None
    out = llm_output.strip()
    # try JSON {"answer": ...}
    try:
        m = re.search(r'\{[^}]*\}', out, re.DOTALL)
        if m:
            d = json.loads(m.group(0))
            ans = str(d.get("answer", "")).strip()
            for opt in valid_options:
                if opt.lower() == ans.lower():
                    return opt
    except Exception:
        pass
    # standalone word match
    for opt in valid_options:
        if re.search(r'\b' + re.escape(opt) + r'\b', out, re.I):
            return opt
    return None


def parse_personas(response_text: str) -> List[dict]:
    """Extract persona list from LLM JSON output."""
    try:
        m = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group(1))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("personas", [])
        return []
    except Exception:
        return []
