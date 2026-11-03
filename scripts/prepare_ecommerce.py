#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Prepare Harvard e-commerce dataset for persona generation.

Dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YGLYDY
Contains: amazon-purchases.csv (1.8M purchases), survey.csv (5K users with demographics)

Usage:
    python scripts/prepare_ecommerce.py --data_dir data/raw/ecommerce/ --max_users 1000
"""
import argparse, csv, json, sys
from pathlib import Path
from collections import Counter, defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw/ecommerce")
    p.add_argument("--max_users", type=int, default=1000)
    p.add_argument("--min_purchases", type=int, default=10)
    p.add_argument("--max_purchases_per_user", type=int, default=50)
    p.add_argument("--output_dir", default="data")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    purchases_file = data_dir / "amazon-purchases.csv"
    survey_file = data_dir / "survey.csv"

    if not purchases_file.exists():
        sys.exit(f"Not found: {purchases_file}\n\nDownload from:\n  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YGLYDY")

    # Load survey demographics
    demographics = {}
    if survey_file.exists():
        with open(survey_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uid = row.get("Survey ResponseID", "")
                if uid:
                    demographics[uid] = {
                        "age": row.get("Q-demos-age", ""),
                        "gender": row.get("Q-demos-gender", ""),
                        "race": row.get("Q-demos-race", ""),
                        "education": row.get("Q-demos-education", ""),
                        "income": row.get("Q-demos-income", ""),
                        "state": row.get("Q-demos-state", ""),
                        "household_size": row.get("Q-amazon-use-hh-size", ""),
                        "order_frequency": row.get("Q-amazon-use-how-oft", ""),
                    }
        print(f"Loaded {len(demographics)} survey respondents")

    # Load purchases grouped by user
    print(f"Loading {purchases_file}...")
    by_user = defaultdict(list)
    with open(purchases_file, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row.get("Survey ResponseID", "")
            if uid:
                by_user[uid].append(row)
    print(f"Loaded purchases for {len(by_user)} users")

    # Randomly sample users with at least min_purchases
    import random
    random.seed(42)
    eligible = [uid for uid, recs in by_user.items() if len(recs) >= args.min_purchases]
    selected = random.sample(eligible, min(args.max_users, len(eligible)))
    sorted_users = [(uid, by_user[uid]) for uid in selected]
    print(f"Selected {len(sorted_users)} users randomly (>= {args.min_purchases} purchases)")

    # Build per-user data
    users = []
    for uid, purchases in sorted_users:
        # Sort by date, take most recent N
        purchases.sort(key=lambda r: r.get("Order Date", ""), reverse=True)
        sample = purchases[:args.max_purchases_per_user]

        responses = []
        for r in sample:
            title = r.get("Title", "Unknown")
            category = r.get("Category", "")
            price = r.get("Purchase Price Per Unit", "")
            date = r.get("Order Date", "")
            state = r.get("Shipping Address State", "")

            q = f"Product: {title}"
            if category:
                q += f" | Category: {category}"
            if price:
                q += f" | Price: ${price}"
            if date:
                q += f" | Date: {date}"

            responses.append({
                "Question": q,
                "Possible_Answers": "",
                "Response": f"Purchased (qty: {r.get('Quantity', '1')})",
            })

        # Compute behavioral features from ALL purchases (summary stats)
        prices = []
        categories = []
        dates = []
        for r in purchases:
            try:
                prices.append(float(r.get("Purchase Price Per Unit", 0)))
            except (ValueError, TypeError):
                pass
            cat = r.get("Category", "")
            if cat:
                categories.append(cat)
            d = r.get("Order Date", "")
            if d:
                dates.append(d)

        feature = demographics.get(uid, {})
        feature["total_purchases"] = str(len(purchases))
        feature["purchases_shown_below"] = str(len(sample))
        if prices:
            feature["avg_price"] = f"${sum(prices)/len(prices):.2f}"
            feature["median_price"] = f"${sorted(prices)[len(prices)//2]:.2f}"
            feature["max_price"] = f"${max(prices):.2f}"
            feature["total_spent"] = f"${sum(prices):.2f}"
        if categories:
            top_cats = Counter(categories).most_common(5)
            feature["top_categories"] = ", ".join(f"{cat} ({n})" for cat, n in top_cats)
        if len(dates) >= 2:
            from datetime import datetime
            try:
                sorted_dates = sorted(dates)
                first = datetime.strptime(sorted_dates[0], "%Y-%m-%d")
                last = datetime.strptime(sorted_dates[-1], "%Y-%m-%d")
                span_days = (last - first).days or 1
                feature["timespan_days"] = str(span_days)
                feature["purchases_per_month"] = f"{len(purchases) / (span_days / 30):.1f}"
            except (ValueError, TypeError):
                pass

        users.append({"id": f"ecom_{uid}", "feature": feature, "responses": responses})

    # Benchmark: use survey questions as ground truth
    # The survey has opinion questions (Q-sell-YOUR-data, Q-sell-consumer-data, etc.)
    # Build benchmark from survey responses
    survey_questions = {
        "Q-sell-YOUR-data": "Would you sell YOUR personal data?",
        "Q-sell-consumer-data": "Should companies be allowed to sell consumer data?",
        "Q-small-biz-use": "Do you use Amazon to support small businesses?",
        "Q-census-use": "Should purchase data be used for census purposes?",
        "Q-research-society": "Should purchase data be used for research to benefit society?",
    }

    questions = []
    if survey_file.exists():
        # Aggregate survey responses across all respondents
        all_survey = []
        with open(survey_file, encoding="utf-8") as f:
            all_survey = list(csv.DictReader(f))

        for qid, qtext in survey_questions.items():
            responses_col = [r.get(qid, "") for r in all_survey if r.get(qid, "")]
            counts = Counter(responses_col)
            if len(counts) < 2:
                continue
            options = {str(i+1): opt for i, opt in enumerate(sorted(counts.keys()))}
            human_counts = {str(i+1): counts[opt] for i, opt in enumerate(sorted(counts.keys()))}
            questions.append({
                "question_id": qid,
                "input_template": qtext,
                "answer_options": options,
                "human_counts": human_counts,
            })

    # Save
    out = Path(args.output_dir)
    user_path = out / "sample" / "ecommerce_users.json"
    bench_path = out / "benchmarks" / "ecommerce.json"
    user_path.parent.mkdir(parents=True, exist_ok=True)
    bench_path.parent.mkdir(parents=True, exist_ok=True)

    user_path.write_text(json.dumps(users, indent=2, ensure_ascii=False))
    bench_path.write_text(json.dumps({"survey_title": "E-commerce Consumer Opinions", "questions": questions}, indent=2))

    print(f"\n✅ {len(users)} users → {user_path}")
    print(f"✅ {len(questions)} questions → {bench_path}")
    print(f"\nRun: python scripts/generate_personas.py --data {user_path} --dataset ecommerce --model claude4.5 --workers 8")


if __name__ == "__main__":
    main()
