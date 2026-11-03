# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Persona induction prompt for SocSci210 social science experiments."""
from __future__ import annotations
from typing import Tuple
from personagen.prompts._shared import PROFILE_FIELDS_BLOCK

SYSTEM = """You are an AI assistant creating simulation-ready personas from social science experiment responses.
The persona will predict how this individual responds to NEW experiment tasks.

Output valid JSON with keys: "analysis", "short_bio", "user_profile".
- "short_bio": SECOND-PERSON ("you"), no name. Free-form text capturing psychological traits, decision patterns, and response tendencies.
- "user_profile": structured fields with "value" and "confidence" (high/medium/low).
Do NOT fabricate specifics unsupported by the data."""


def persona_prompt(behavior_data: str, metadata: str = "") -> Tuple[str, str]:
    demo_block = f"\n<demographics>\n{metadata}\n</demographics>" if metadata else ""

    user = f"""<user_data>
<behavior_data>
{behavior_data}
</behavior_data>{demo_block}
</user_data>

Create a simulation-ready persona from this participant's experiment responses.

1. "analysis": Identify stable psychological traits, moral reasoning, risk preferences, social attitudes, and decision-making style.
2. "short_bio": Free-form persona in second-person. Focus on predictive traits, decision rules, attitude strength, cognitive style.
3. "user_profile": Extract ALL of the following fields:
{PROFILE_FIELDS_BLOCK}

Additionally extract domain-specific traits:
- moral_foundations: Primary moral reasoning style [Care, Fairness, Loyalty, Authority, Purity, Liberty]
- risk_tolerance: [Risk-averse, Moderate, Risk-seeking]
- social_trust: [High, Moderate, Low]
- cognitive_style: [Analytical, Intuitive, Balanced]

For each field, include "value" and "confidence" (high/medium/low).

Output JSON only."""

    return SYSTEM, user
