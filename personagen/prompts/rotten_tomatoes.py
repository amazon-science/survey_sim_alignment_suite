# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Persona induction prompt for Rotten Tomatoes movie critics."""
from __future__ import annotations
from typing import Tuple
from personagen.prompts._shared import PROFILE_FIELDS_BLOCK

SYSTEM = """You are an AI assistant creating simulation-ready personas from movie critic review histories.
The persona will predict how this critic evaluates NEW films.

Output valid JSON with keys: "analysis", "short_bio", "user_profile".
- "short_bio": SECOND-PERSON ("you"), no name. Free-form text capturing taste profile, evaluation criteria, and rating tendencies.
- "user_profile": structured fields with "value" and "confidence" (high/medium/low).
Do NOT fabricate specifics unsupported by the data."""


def persona_prompt(behavior_data: str, metadata: str = "") -> Tuple[str, str]:
    meta_block = f"\n<metadata>\n{metadata}\n</metadata>" if metadata else ""

    user = f"""<user_data>
<review_history>
{behavior_data}
</review_history>{meta_block}
</user_data>

Create a simulation-ready persona from this critic's review history.

1. "analysis": Identify taste patterns, genre preferences, quality thresholds, harshness/generosity.
2. "short_bio": Free-form persona in second-person. Focus on what drives Fresh vs Rotten verdicts, aesthetic preferences, evaluation priorities.
3. "user_profile": Extract ALL of the following fields:
{PROFILE_FIELDS_BLOCK}

Additionally extract domain-specific traits:
- rating_tendency: [Generous, Balanced, Harsh]
- preferred_genres: Top genre affinities
- evaluation_priorities: What matters most [Acting, Direction, Writing, Visuals, Originality, Emotional impact]
- audience_perspective: [Cinephile, General audience, Genre specialist]
- fresh_rate: Approximate positive review rate [High (>70%), Medium (50-70%), Low (<50%)]

For each field, include "value" and "confidence" (high/medium/low).

Output JSON only."""

    return SYSTEM, user
