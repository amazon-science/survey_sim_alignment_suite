#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Generate a persona pool from per-user survey response data.

Induces one persona per user from their survey responses (Free prompt).
Outputs a flat list of personas for population-level simulation.

Usage:
    python scripts/generate_personas.py \
        --data data/sample/socsci210_mp3dr_users.json \
        --dataset socsci210 --model claude4.5 --workers 8
"""

from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from personagen.prompts import persona_induction_prompt
from personagen.llm import call_bedrock_parallel


def _format_behavior_data(user: dict, dataset: str) -> str:
    """Format user behavioral data based on dataset domain."""
    if dataset == "rotten_tomatoes":
        lines = []
        for i, r in enumerate(user["responses"]):
            if not r.get("Response"):
                continue
            lines.append(f"Review {i+1}: {r['Question']}")
            lines.append(f"Verdict: {r['Response']}")
        return "\n".join(lines)
    elif dataset == "ecommerce":
        lines = []
        for i, r in enumerate(user["responses"]):
            if not r.get("Response"):
                continue
            lines.append(f"Purchase {i+1}: {r['Question']}")
            lines.append(f"Rating/Review: {r['Response']}")
        return "\n".join(lines)
    else:  # socsci210 and other survey-type datasets
        lines = []
        for i, r in enumerate(user["responses"]):
            if not r.get("Response"):
                continue
            lines.append(f"Task {i+1}: {r['Question']}")
            if r.get("Possible_Answers"):
                lines.append(f"Options: {r['Possible_Answers']}")
            lines.append(f"Response: {r['Response']}")
        return "\n".join(lines)


def _format_metadata(user: dict) -> str:
    """Format user feature/demographic data as metadata string."""
    feature = user.get("feature", {})
    if not feature:
        return ""
    return "\n".join(f"- {k}: {v}" for k, v in feature.items() if v)


def _parse_persona_response(text: str) -> dict | None:
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if not m:
            return None
        data = json.loads(m.group(0))
        if "short_bio" not in data:
            return None
        return data
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(description="Generate persona pool from per-user survey data")
    p.add_argument("--data", required=True, help="Per-user JSON with responses")
    p.add_argument("--dataset", default="gss")
    p.add_argument("--model", default="claude3.7")
    p.add_argument("--max_tokens", type=int, default=50000)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--profile", default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    users = json.loads(Path(args.data).read_text())
    print(f"Loaded {len(users)} users from {args.data}")

    out_path = Path(args.output) if args.output else Path("output") / f"{args.dataset}_personas.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_path.with_suffix(".checkpoint.json")

    # Resume from checkpoint if exists
    completed = {}
    if checkpoint_path.exists():
        completed = {p["user_id"]: p for p in json.loads(checkpoint_path.read_text())}
        print(f"♻️  Resuming: {len(completed)} personas already done")

    # Filter to only users not yet completed
    remaining = [(i, u) for i, u in enumerate(users) if u.get("id", "") not in completed]
    print(f"Remaining: {len(remaining)} users to process")

    if remaining:
        prompts = []
        for _, u in remaining:
            text = _format_behavior_data(u, args.dataset)
            metadata = _format_metadata(u)
            sys_p, usr_p = persona_induction_prompt(text, dataset=args.dataset, metadata=metadata)
            prompts.append((sys_p, usr_p))

        print(f"Calling Bedrock ({len(prompts)} requests, model={args.model}) …")
        responses = call_bedrock_parallel(
            prompts, args.model,
            max_tokens=args.max_tokens, temperature=args.temperature,
            region=args.region, profile=args.profile, workers=args.workers,
            on_result=lambda idx, txt: _save_checkpoint(
                checkpoint_path, completed, remaining[idx][1], txt
            ),
        )

        # Parse any remaining responses not yet checkpointed
        for (_, user), resp in zip(remaining, responses):
            uid = user.get("id", "")
            if uid not in completed:
                _save_checkpoint(checkpoint_path, completed, user, resp)

    # Final output from checkpoint
    personas = list(completed.values())
    print(f"✅ {len(personas)}/{len(users)} personas")
    out_path.write_text(json.dumps(personas, indent=2, ensure_ascii=False))
    print(f"💾 Saved to {out_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("🧹 Checkpoint removed")


def _save_checkpoint(checkpoint_path, completed, user, resp):
    """Parse response and append to checkpoint."""
    parsed = _parse_persona_response(resp)
    if not parsed:
        return
    uid = user.get("id", "")
    persona = {
        "short_bio": parsed.get("short_bio", ""),
        "user_profile": parsed.get("user_profile", {}),
        "user_id": uid,
    }
    completed[uid] = persona
    checkpoint_path.write_text(json.dumps(list(completed.values()), ensure_ascii=False))


if __name__ == "__main__":
    main()
