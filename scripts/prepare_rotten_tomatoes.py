#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Prepare Rotten Tomatoes critic reviews for persona generation.

Requires Kaggle download:
    kaggle datasets download stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset
    unzip rotten-tomatoes-movies-and-critic-reviews-dataset.zip -d data/raw/rotten_tomatoes/

Usage:
    python scripts/prepare_rotten_tomatoes.py --data_dir data/raw/rotten_tomatoes/
"""
import argparse, csv, json, sys
from pathlib import Path
from collections import Counter, defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw/rotten_tomatoes")
    p.add_argument("--max_users", type=int, default=50)
    p.add_argument("--min_reviews", type=int, default=20)
    p.add_argument("--output_dir", default="data")
    args = p.parse_args()

    reviews_file = Path(args.data_dir) / "rotten_tomatoes_critic_reviews.csv"
    if not reviews_file.exists():
        sys.exit(f"Not found: {reviews_file}\n\nDownload:\n  kaggle datasets download stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset\n  unzip *.zip -d {args.data_dir}")

    # Load movie metadata
    movies_file = Path(args.data_dir) / "rotten_tomatoes_movies.csv"
    movies = {}
    if movies_file.exists():
        with open(movies_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                link = row.get("rotten_tomatoes_link", "")
                if link:
                    movies[link] = row
        print(f"Loaded {len(movies)} movies")

    print(f"Loading {reviews_file}...")
    reviews = []
    with open(reviews_file, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("critic_name") and row.get("review_type"):
                # Attach movie info
                link = row.get("rotten_tomatoes_link", "")
                movie = movies.get(link, {})
                row["movie_title"] = movie.get("movie_title", link)
                row["genres"] = movie.get("genres", "")
                row["directors"] = movie.get("directors", "")
                row["content_rating"] = movie.get("content_rating", "")
                row["tomatometer_rating"] = movie.get("tomatometer_rating", "")
                row["audience_rating"] = movie.get("audience_rating", "")
                row["actors"] = movie.get("actors", "")
                row["release_year"] = movie.get("original_release_date", "")[:4]
                reviews.append(row)
    print(f"Loaded {len(reviews)} reviews")

    # Group by critic, filter, take top N
    by_critic = defaultdict(list)
    for r in reviews:
        by_critic[r["critic_name"]].append(r)

    critics = sorted(
        ((name, revs) for name, revs in by_critic.items() if len(revs) >= args.min_reviews),
        key=lambda x: -len(x[1])
    )[:args.max_users]

    users = []
    for critic_name, revs in critics:
        responses = []
        for r in revs[:30]:
            movie = r.get("movie_title", "Unknown")
            genres = r.get("genres", "")
            directors = r.get("directors", "")
            actors = r.get("actors", "")
            year = r.get("release_year", "")
            tomatometer = r.get("tomatometer_rating", "")
            audience = r.get("audience_rating", "")
            content = (r.get("review_content") or "")[:200]
            q = f"Movie: {movie}"
            if year:
                q += f" ({year})"
            if genres:
                q += f" | Genre: {genres}"
            if directors:
                q += f" | Director: {directors}"
            if actors:
                q += f" | Actors: {actors}"
            if tomatometer:
                q += f" | Tomatometer: {tomatometer}%"
            if audience:
                q += f" | Audience: {audience}%"
            if content:
                q += f" | Review: {content}"
            responses.append({"Question": q, "Possible_Answers": "Fresh; Rotten", "Response": r["review_type"]})
        feature = {"critic_name": critic_name}
        if revs[0].get("publisher_name"):
            feature["publisher"] = revs[0]["publisher_name"]
        users.append({"id": f"rt_{critic_name.replace(' ', '_')[:30]}", "feature": feature, "responses": responses})

    # Benchmark: sentiment distribution per top movie
    movie_sentiments = defaultdict(list)
    for r in reviews:
        if r.get("movie_title") and r.get("review_type"):
            movie_sentiments[r["movie_title"]].append(r["review_type"])

    questions = []
    for i, (movie, sents) in enumerate(sorted(movie_sentiments.items(), key=lambda x: -len(x[1]))[:100]):
        counts = Counter(sents)
        if len(counts) < 2:
            continue
        questions.append({
            "question_id": str(i),
            "input_template": f"What is your overall assessment of the movie '{movie}'?",
            "answer_options": {"1": "Fresh", "2": "Rotten"},
            "human_counts": {"1": counts.get("Fresh", 0), "2": counts.get("Rotten", 0)},
        })

    out = Path(args.output_dir)
    user_path = out / "sample" / "rotten_tomatoes_users.json"
    bench_path = out / "benchmarks" / "rotten_tomatoes.json"
    user_path.parent.mkdir(parents=True, exist_ok=True)
    bench_path.parent.mkdir(parents=True, exist_ok=True)

    user_path.write_text(json.dumps(users, indent=2, ensure_ascii=False))
    bench_path.write_text(json.dumps({"survey_title": "Rotten Tomatoes Critic Opinions", "questions": questions}, indent=2))

    print(f"✅ {len(users)} users → {user_path}")
    print(f"✅ {len(questions)} questions → {bench_path}")
    print(f"\nRun: DATA_USERS={user_path} DATA_BENCHMARK={bench_path} DATASET=rotten_tomatoes bash run.sh")


if __name__ == "__main__":
    main()
