# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Survey simulation prompt — shared across all domains."""
from __future__ import annotations
from typing import Dict, Tuple

SYSTEM_PROMPT = (
    "You are an AI assistant tasked with role-playing a person with specific characteristics. "
    "You are answering a survey. You are provided with a multiple choice question and you have "
    "to select the answer option based on the character you are playing."
)

PROMPT_TEMPLATE = """Follow these instructions carefully:

1. You will be given some background information about the character inside <background> tags.
<background>
{persona}
</background>

2. Your task is to choose 1 option from the list and provide a reason for your choice, based on the background information.

3. Your response should be in valid JSON format, containing two keys: "answer" (the chosen option number) and "reason" (the explanation for your choice).

4. Important notes:
- Only choose from the options provided.
- Ensure your reason aligns with the given background information.
- Do not include any additional text outside the JSON object.

You are presented with a question inside <question> tags.
<question>
{question}
</question>

<options>
{options}
</options>"""


def survey_inference_prompt(
    question_text: str,
    options: Dict[str, str],
    persona_bio: str,
    dataset: str = "",
) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for single-choice survey simulation."""
    opts_str = "\n".join(f"{k}: {v}" for k, v in options.items())
    user_prompt = PROMPT_TEMPLATE.format(
        persona=persona_bio,
        question=question_text,
        options=opts_str,
    )
    return SYSTEM_PROMPT, user_prompt
