#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Prepare SocSci210 dataset for persona generation.

Downloads from HuggingFace: socratesft/SocSci210
Outputs per-user behavioral data + benchmark file.

Usage:
    pip install datasets
    python scripts/prepare_socsci210.py --study_id 9nphm --max_users 50
    python scripts/prepare_socsci210.py --list_studies
"""
import argparse, json, sys
from pathlib import Path
from collections import Counter, defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study_id", default="9nphm")
    p.add_argument("--max_users", type=int, default=50)
    p.add_argument("--list_studies", action="store_true")
    p.add_argument("--output_dir", default="data")
    args = p.parse_args()

    from datasets import load_dataset
    print("Loading SocSci210 from HuggingFace...")
    ds = load_dataset("socratesft/SocSci210", split="train")

    if args.list_studies:
        studies = Counter(row["study_id"] for row in ds)
        for sid, count in studies.most_common(30):
            print(f"{sid:<12} {count:>8} rows")
        return

    rows = [r for r in ds if r["study_id"] == args.study_id]
    if not rows:
        sys.exit(f"Study '{args.study_id}' not found.")
    print(f"Study {args.study_id}: {len(rows)} rows")

    by_participant = defaultdict(list)
    for r in rows:
        by_participant[r["participant"]].append(r)

    # Select participants with the most responses to ensure rich persona context
    pids = sorted(by_participant.keys(), key=lambda pid: -len(by_participant[pid]))[:args.max_users]
    print(f"Using top {len(pids)} participants by response count (min={len(by_participant[pids[-1]])}, max={len(by_participant[pids[0]])})")

    users = []
    for pid in pids:
        recs = by_participant[pid]
        demo = recs[0].get("demographic") or {}
        feature = {k: str(v) for k, v in demo.items() if v is not None}
        responses = [
            {"Question": r.get("stimuli", ""), "Possible_Answers": "Integer scale", "Response": str(r["response"])}
            for r in recs
        ]
        users.append({"id": f"socsci_{args.study_id}_{pid}", "feature": feature, "responses": responses})

    # Benchmark
    task_responses = defaultdict(list)
    task_stimuli = {}
    for r in rows:
        key = (r["task_num"], r["condition_num"])
        task_responses[key].append(r["response"])
        task_stimuli[key] = r.get("stimuli", "")

    questions = []
    for i, (key, resps) in enumerate(sorted(task_responses.items())):
        counts = Counter(resps)
        questions.append({
            "question_id": str(i),
            "input_template": task_stimuli[key],
            "answer_options": {str(k): str(k) for k in sorted(counts.keys())},
            "human_counts": {str(k): v for k, v in counts.items()},
        })

    out = Path(args.output_dir)
    name = f"socsci210_{args.study_id}"
    user_path = out / "sample" / f"{name}_users.json"
    bench_path = out / "benchmarks" / f"{name}.json"
    user_path.parent.mkdir(parents=True, exist_ok=True)
    bench_path.parent.mkdir(parents=True, exist_ok=True)

    user_path.write_text(json.dumps(users, indent=2, ensure_ascii=False))
    bench_path.write_text(json.dumps({"survey_title": f"SocSci210 - {args.study_id}", "questions": questions}, indent=2))

    print(f"✅ {len(users)} users → {user_path}")
    print(f"✅ {len(questions)} questions → {bench_path}")
    print(f"\nRun: DATA_USERS={user_path} DATA_BENCHMARK={bench_path} DATASET=socsci210 bash run.sh")


if __name__ == "__main__":
    main()
