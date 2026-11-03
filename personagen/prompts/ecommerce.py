# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Persona induction prompt for e-commerce customers.

Persona structure: demographic background, shopping behavior, spending patterns,
quality preferences.
"""
from __future__ import annotations
from typing import Tuple
from personagen.prompts._shared import PROFILE_FIELDS_BLOCK

SYSTEM = """You are an AI assistant creating simulation-ready personas from e-commerce purchase histories.
The persona will predict how this customer responds to product surveys and shopping preference questions.

Output valid JSON with keys: "analysis", "short_bio", "user_profile".
- "short_bio": SECOND-PERSON ("you"), no name. Free-form text capturing shopping psychology, decision criteria, brand attitudes, and preference patterns.
- "user_profile": structured fields with "value" and "confidence" (high/medium/low).
Do NOT fabricate specifics unsupported by the data."""


def persona_prompt(behavior_data: str, metadata: str = "") -> Tuple[str, str]:
    meta_block = f"\n<metadata>\n{metadata}\n</metadata>" if metadata else ""

    user = f"""<user_data>
<purchase_history>
{behavior_data}
</purchase_history>{meta_block}
</user_data>

Create a simulation-ready persona from this customer's purchase/review history.
Note: The metadata summarizes the customer's FULL purchase history. The individual purchase records above are a representative sample.

1. "analysis": Identify shopping patterns, price sensitivity, brand loyalty, quality expectations, category preferences.
2. "short_bio": Free-form persona in second-person. Focus on shopping decision style, what drives satisfaction, brand attitudes, price/quality tradeoffs.
3. "user_profile": Extract ALL of the following fields:
{PROFILE_FIELDS_BLOCK}

Additionally extract domain-specific traits:
- price_sensitivity: [Very price-sensitive, Moderate, Price-insensitive]
- brand_loyalty: [Brand-loyal, Flexible, Deal-seeking]
- shopping_frequency: [Heavy (weekly+), Regular (monthly), Occasional]
- quality_focus: [Premium-seeking, Balanced, Budget-focused]
- decision_style: [Research-heavy, Impulsive, Recommendation-driven]
- top_categories: Primary shopping categories

For each field, include "value" and "confidence" (high/medium/low).

Output JSON only."""

    return SYSTEM, user
