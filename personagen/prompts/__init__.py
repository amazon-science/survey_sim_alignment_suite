# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Domain-specific prompts for persona induction and survey simulation."""

from personagen.prompts.socsci210 import persona_prompt as socsci210_persona_prompt
from personagen.prompts.rotten_tomatoes import persona_prompt as rotten_tomatoes_persona_prompt
from personagen.prompts.ecommerce import persona_prompt as ecommerce_persona_prompt
from personagen.prompts.simulation import survey_inference_prompt

PERSONA_PROMPTS = {
    "socsci210": socsci210_persona_prompt,
    "rotten_tomatoes": rotten_tomatoes_persona_prompt,
    "ecommerce": ecommerce_persona_prompt,
}


def persona_induction_prompt(behavior_data: str, dataset: str, metadata: str = ""):
    """Route to domain-specific persona induction prompt."""
    fn = PERSONA_PROMPTS.get(dataset, socsci210_persona_prompt)
    return fn(behavior_data, metadata)
