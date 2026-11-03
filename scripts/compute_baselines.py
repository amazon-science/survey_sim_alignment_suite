#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Compute significance tests: for each benchmark×metric, find the best performer
(across persona sources + baselines), then mark all others with * if significantly different."""
import json, sys, numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from personagen.evaluation import compute_alignment, parse_choice

BENCHMARKS = ['gss', 'opinionqa_w34', 'opinionqa_w50', 'opinionqa_w92']
SOURCES = ['socsci210', 'rotten_tomatoes', 'ecommerce']
BASELINES = ['normal', 'random', 'uniform']
METRICS = ['TRM', 'RC', 'TVD', 'DH', 'CV', 'JSD', 'WD']
HIGHER_BETTER = {'TRM': True, 'RC': True, 'TVD': False, 'DH': True, 'CV': False, 'JSD': False, 'WD': False}

np.random.seed(42)


def generate_baseline(human_counts, n_agents, method):
    k = len(human_counts)
    if method == 'uniform':
        counts = [n_agents // k] * k
        for i in range(n_agents % k):
            counts[i] += 1
    elif method == 'normal':
        probs = np.exp(-0.5 * ((np.arange(k) - (k - 1) / 2) / max(k / 4, 1)) ** 2)
        probs /= probs.sum()
        counts = np.random.multinomial(n_agents, probs).tolist()
    elif method == 'random':
        probs = np.random.dirichlet(np.ones(k))
        counts = np.random.multinomial(n_agents, probs).tolist()
    return counts


def get_persona_scores(source, bench):
    inf_path = Path(f'output/{source}_{bench}_inference.json')
    questions = json.loads(inf_path.read_text())
    scores = []
    for q in questions:
        raw = q.get('agent_responses', [])
        if not raw:
            continue
        keys = list(q['human_counts'].keys())
        human_counts = [q['human_counts'][k] for k in keys]
        agent_counts = [0] * len(keys)
        for resp in raw:
            choice = parse_choice(resp, keys)
            if choice and choice in keys:
                agent_counts[keys.index(choice)] += 1
        if sum(agent_counts) == 0:
            continue
        scores.append(compute_alignment(human_counts, agent_counts))
    return scores


def main():
    out_dir = Path('output/baselines')
    out_dir.mkdir(parents=True, exist_ok=True)

    for bench in BENCHMARKS:
        bench_data = json.loads(Path(f'data/benchmarks/{bench}.json').read_text())
        questions = bench_data['questions']

        # Collect per-question scores for all conditions
        all_scores = {}
        for src in SOURCES:
            all_scores[src] = get_persona_scores(src, bench)
        n_q = len(all_scores[SOURCES[0]])

        for method in BASELINES:
            baseline_scores = []
            for q in questions:
                keys = list(q['human_counts'].keys())
                human_counts = [q['human_counts'][k] for k in keys]
                if sum(human_counts) == 0:
                    continue
                agent_counts = generate_baseline(human_counts, 1000, method)
                baseline_scores.append(compute_alignment(human_counts, agent_counts))
            if len(baseline_scores) == n_q:
                all_scores[method] = baseline_scores
            # Save
            save_path = out_dir / f'{bench}_{method}_per_question.json'
            save_path.write_text(json.dumps(baseline_scores, indent=2))

        # For each metric, find best condition, then test all others against it
        for m in METRICS:
            means = {cond: np.mean([s[m] for s in scores]) for cond, scores in all_scores.items()}
            if HIGHER_BETTER[m]:
                best_cond = max(means, key=means.get)
            else:
                best_cond = min(means, key=means.get)
            best_vals = [s[m] for s in all_scores[best_cond]]

            sig_map = {}
            for cond, scores in all_scores.items():
                if cond == best_cond:
                    sig_map[cond] = ''
                else:
                    _, p = stats.ttest_rel(best_vals, [s[m] for s in scores])
                    sig_map[cond] = '*' if p < 0.05 else ''

            # Print
            print(f"{bench:<14} {m:<4} best={best_cond:<16} | " +
                  " | ".join(f"{c}: {means[c]:.3f}{sig_map[c]}" for c in SOURCES + BASELINES if c in all_scores))

        print()

    # Save full results as JSON
    # Re-run to build exportable structure
    results = {}
    for bench in BENCHMARKS:
        bench_data = json.loads(Path(f'data/benchmarks/{bench}.json').read_text())
        questions = bench_data['questions']
        all_scores = {}
        for src in SOURCES:
            all_scores[src] = get_persona_scores(src, bench)
        n_q = len(all_scores[SOURCES[0]])
        for method in BASELINES:
            path = out_dir / f'{bench}_{method}_per_question.json'
            if path.exists():
                scores = json.loads(path.read_text())
                if len(scores) == n_q:
                    all_scores[method] = scores

        bench_results = {}
        for m in METRICS:
            means = {c: float(np.mean([s[m] for s in sc])) for c, sc in all_scores.items()}
            if HIGHER_BETTER[m]:
                best_cond = max(means, key=means.get)
            else:
                best_cond = min(means, key=means.get)
            best_vals = [s[m] for s in all_scores[best_cond]]

            metric_results = {}
            for cond, scores in all_scores.items():
                if cond == best_cond:
                    metric_results[cond] = {'mean': means[cond], 'sig': False, 'is_best': True}
                else:
                    _, p = stats.ttest_rel(best_vals, [s[m] for s in scores])
                    metric_results[cond] = {'mean': means[cond], 'p': float(p), 'sig': bool(p < 0.05), 'is_best': False}
            bench_results[m] = {'best': best_cond, 'conditions': metric_results}
        results[bench] = bench_results

    summary_path = out_dir / 'significance_summary.json'
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_dir}/")


if __name__ == '__main__':
    main()
