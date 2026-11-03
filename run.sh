#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

set -euo pipefail

# ── Configuration ──────────────────────────────────────────
MODEL="${MODEL:-claude4.5}"
DATASET="${DATASET:-socsci210}"
STUDY_ID="${STUDY_ID:-}"
WORKERS="${WORKERS:-8}"
REGION="${REGION:-us-east-1}"
PROFILE="${PROFILE:-}"

# Build data file suffix (e.g. socsci210_9nphm or just socsci210)
NAME="${DATASET}"
[ -n "$STUDY_ID" ] && NAME="${DATASET}_${STUDY_ID}"

DATA_USERS="${DATA_USERS:-data/sample/${NAME}_users.json}"
DATA_BENCHMARK="${DATA_BENCHMARK:-data/benchmarks/${NAME}.json}"
PERSONA_FILE="output/${NAME}_personas.json"
# ───────────────────────────────────────────────────────────

PROFILE_FLAG=""
[ -n "$PROFILE" ] && PROFILE_FLAG="--profile $PROFILE"

echo "═══════════════════════════════════════════"
echo " PersonaGen Pipeline"
echo " Model: $MODEL  Dataset: $DATASET"
echo "═══════════════════════════════════════════"

echo ""
echo "▶ Step 1: Induce personas from behavioral data"
python scripts/generate_personas.py \
    --data "$DATA_USERS" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --workers "$WORKERS" \
    --region "$REGION" \
    $PROFILE_FLAG

echo ""
echo "▶ Step 2: Run simulation & evaluate with RADIUS"
python scripts/run_inference.py \
    --data "$DATA_BENCHMARK" \
    --personas "$PERSONA_FILE" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --workers "$WORKERS" \
    --region "$REGION" \
    $PROFILE_FLAG

echo ""
echo "═══════════════════════════════════════════"
echo " ✅ Done — results in output/"
echo "═══════════════════════════════════════════"
