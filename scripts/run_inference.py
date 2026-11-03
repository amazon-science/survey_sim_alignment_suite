#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Run population-level survey simulation and evaluate alignment with RADIUS.

Every persona in the pool answers every question. Responses are aggregated
into one distribution per question and compared against human counts.

Usage:
    python scripts/run_inference.py \
        --data data/benchmarks/gss.json \
        --personas output/socsci210_personas.json \
        --dataset socsci210 --model claude4.5
"""

from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from personagen.prompts.simulation import survey_inference_prompt
from personagen.evaluation import compute_alignment, parse_choice, METRICS
from personagen.llm import call_bedrock_parallel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Survey JSON with human_counts per question")
    p.add_argument("--personas", required=True, help="Persona pool JSON from generate_personas.py")
    p.add_argument("--dataset", default="gss")
    p.add_argument("--model", default="claude4.5")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--profile", default=None)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--max_personas", type=int, default=None, help="Limit number of personas")
    p.add_argument("--batch", action="store_true", help="Use batch inference (50%% cheaper)")
    p.add_argument("--s3_bucket", default=None, help="S3 bucket for batch I/O")
    p.add_argument("--role_arn", default=None, help="IAM role ARN for batch job")
    return p.parse_args()


def main():
    args = parse_args()
    raw = json.loads(Path(args.data).read_text())
    personas = json.loads(Path(args.personas).read_text())

    # Build output name: {persona_source}_{benchmark}
    benchmark_name = Path(args.data).stem  # e.g. "global_social_survey_gss"
    run_name = f"{args.dataset}_{benchmark_name}"

    if args.max_personas:
        personas = personas[:args.max_personas]

    # Support both flat list and benchmark wrapper format
    questions = raw["questions"] if isinstance(raw, dict) and "questions" in raw else raw

    print(f"Questions: {len(questions)}  |  Personas: {len(personas)}  |  Mode: {'batch' if args.batch else 'on-demand'}")

    # ---- Inference ----
    if args.eval_only:
        # Load previously saved inference results
        inter = Path(args.output or f"output/{run_name}_inference.json")
        if inter.exists():
            questions = json.loads(inter.read_text())
            print(f"📂 Loaded inference from {inter}")
    else:
        all_prompts = []
        prompt_map = []  # (question_idx, persona_idx)

        for qi, q in enumerate(questions):
            options = q.get("answer_options", {})
            if not options:
                options = {k: k for k in q["human_counts"].keys()}
            for pi, persona in enumerate(personas):
                bio = persona.get("short_bio", "")
                sys_p, usr_p = survey_inference_prompt(
                    q["input_template"], options, bio, args.dataset,
                )
                all_prompts.append((sys_p, usr_p))
                prompt_map.append((qi, pi))

        print(f"Running inference: {len(all_prompts)} prompts ({len(questions)} questions × {len(personas)} personas) …")

        inter = Path(args.output or f"output/{run_name}_inference.json")
        inter.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = inter.with_suffix(".checkpoint.json")

        # Resume from checkpoint if exists
        responses = [None] * len(all_prompts)
        start_from = 0
        if checkpoint.exists():
            ckpt = json.loads(checkpoint.read_text())
            # Validate checkpoint matches current run
            if (ckpt.get("total") == len(all_prompts)
                and ckpt.get("data") == args.data
                and ckpt.get("num_personas") == len(personas)):
                for idx_str, resp in ckpt.get("results", {}).items():
                    responses[int(idx_str)] = resp
                start_from = sum(1 for r in responses if r is not None)
                print(f"♻️  Resuming from checkpoint: {start_from}/{len(all_prompts)} done")
            else:
                print("⚠️  Checkpoint doesn't match current run, starting fresh")

        def _save_checkpoint(idx, txt):
            responses[start_from + idx] = txt
            if (start_from + idx + 1) % 50 == 0:
                _write_checkpoint()

        def _write_checkpoint():
            ckpt = {
                "data": args.data,
                "num_personas": len(personas),
                "total": len(all_prompts),
                "results": {str(i): r for i, r in enumerate(responses) if r is not None},
            }
            checkpoint.write_text(json.dumps(ckpt, ensure_ascii=False))

        if start_from < len(all_prompts):
            remaining_prompts = all_prompts[start_from:]

            if args.batch:
                if not args.s3_bucket or not args.role_arn:
                    sys.exit("--batch requires --s3_bucket and --role_arn")
                from personagen.llm import call_bedrock_batch
                batch_results = call_bedrock_batch(
                    remaining_prompts, args.model,
                    s3_bucket=args.s3_bucket, max_tokens=args.max_tokens,
                    temperature=args.temperature, region=args.region,
                    profile=args.profile, role_arn=args.role_arn,
                )
                for i, r in enumerate(batch_results):
                    responses[start_from + i] = r
            else:
                partial = call_bedrock_parallel(
                    remaining_prompts, args.model,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    region=args.region, profile=args.profile, workers=args.workers,
                    on_result=_save_checkpoint,
                )
                for i, r in enumerate(partial):
                    responses[start_from + i] = r

            _write_checkpoint()

        # Attach responses
        for q in questions:
            q["agent_responses"] = []
        for (qi, pi), resp in zip(prompt_map, responses):
            if resp is not None:
                questions[qi]["agent_responses"].append(resp)

        inter.write_text(json.dumps(questions, indent=2, ensure_ascii=False))
        print(f"💾 Inference saved to {inter}")

        # Clean up checkpoint
        if checkpoint.exists():
            checkpoint.unlink()

    # ---- Evaluation ----
    print("\nEvaluating …")
    all_metrics = defaultdict(list)
    question_rows = [["question_id", "question", "human_counts", "agent_counts"] + list(METRICS.keys())]

    for q in tqdm(questions, desc="Eval"):
        raw = q.get("agent_responses", [])
        if not raw:
            continue

        keys = list(q["human_counts"].keys())
        human_counts = [q["human_counts"][k] for k in keys]

        # Aggregate agent option selections into counts
        agent_counts = [0] * len(keys)
        for resp in raw:
            choice = parse_choice(resp, keys)
            if choice and choice in keys:
                agent_counts[keys.index(choice)] += 1

        if sum(agent_counts) == 0:
            continue

        metrics = compute_alignment(human_counts, agent_counts)
        for name, val in metrics.items():
            all_metrics[name].append(val)

        question_rows.append([
            q.get("question_id", ""),
            q.get("input_template", "")[:100],
            json.dumps(dict(zip(keys, human_counts))),
            json.dumps(dict(zip(keys, agent_counts))),
        ] + [f"{metrics[m]:.4f}" for m in METRICS.keys()])

    # Print results
    print("\n" + "=" * 65)
    print(f"{'Metric':<28} {'Mean':>10} {'Std':>10} {'N':>6}")
    print("=" * 65)
    rows = [["Metric", "Mean", "Std", "N"]]
    for name in all_metrics:
        vals = all_metrics[name]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        rows.append([name, f"{mean:.4f}", f"{std:.4f}", str(len(vals))])
        print(f"{name:<28} {mean:>10.4f} {std:>10.4f} {len(vals):>6}")

    csv_path = Path(f"output/{run_name}_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n💾 Results CSV: {csv_path}")

    # Per-question results
    q_csv_path = csv_path.with_name(csv_path.stem + "_per_question.csv")
    with open(q_csv_path, "w", newline="") as f:
        csv.writer(f).writerows(question_rows)
    print(f"💾 Per-question CSV: {q_csv_path}")


if __name__ == "__main__":
    main()
