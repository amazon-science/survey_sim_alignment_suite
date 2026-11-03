# Persona Generation & Survey Simulation Pipeline

This document describes the simulation pipeline used to validate RADIUS across multiple persona sources and survey benchmarks.

## Prerequisites

```bash
pip install -r requirements.txt
# AWS credentials for Bedrock batch inference
aws configure --profile <YOUR_PROFILE>
```

## Step 1: Download raw data

- **SocSci210**: Auto-downloaded by the prepare script from [HuggingFace](https://huggingface.co/datasets/socratesft/SocSci210)
- **Rotten Tomatoes**: Download from [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) → `data/raw/rotten_tomatoes/`
- **E-commerce**: Download from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YGLYDY) → `data/raw/ecommerce/`

See [data/README.md](../data/README.md) for full details on all data sources and benchmark preparation.

## Step 2: Prepare per-user data

```bash
python scripts/prepare_socsci210.py --study_id mp3dr --max_users 1000
python scripts/prepare_rotten_tomatoes.py --data_dir data/raw/rotten_tomatoes/ --max_users 1000
python scripts/prepare_ecommerce.py --data_dir data/raw/ecommerce/ --max_users 1000
```

## Step 3: Generate personas

```bash
for src in socsci210 rotten_tomatoes ecommerce; do
    python scripts/generate_personas.py \
        --data data/sample/${src}_users.json \
        --dataset $src --model claude4.5 --workers 8 \
        --profile <YOUR_PROFILE>
done
```

## Step 4: Run survey simulations

```bash
for src in socsci210 rotten_tomatoes ecommerce; do
    for bench in gss opinionqa_w34 opinionqa_w50 opinionqa_w92; do
        python scripts/run_inference.py \
            --data data/benchmarks/${bench}.json \
            --personas output/${src}_personas.json \
            --dataset $src --model claude4.5 --batch \
            --s3_bucket <YOUR_S3_BUCKET> \
            --role_arn <YOUR_BEDROCK_ROLE_ARN> \
            --profile <YOUR_PROFILE>
    done
done
```

## Step 5: Evaluate

```bash
for src in socsci210 rotten_tomatoes ecommerce; do
    for bench in gss opinionqa_w34 opinionqa_w50 opinionqa_w92; do
        python scripts/run_inference.py \
            --data data/benchmarks/${bench}.json \
            --personas output/${src}_personas.json \
            --dataset $src --eval_only
    done
done
```

## Step 6: Compute baselines and significance tests

```bash
python scripts/compute_baselines.py
```

## Persona Sources

| Source | Dataset | Users | Behavioral signal |
|--------|---------|-------|-------------------|
| SocSci210 | [HuggingFace](https://huggingface.co/datasets/socratesft/SocSci210) | 999 | Social science survey responses |
| Rotten Tomatoes | [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) | 1000 | Movie critic reviews (Fresh/Rotten) |
| E-commerce | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YGLYDY) | 1000 | Amazon purchase histories + demographics |

## Benchmark Datasets

| Dataset | Questions | Topic |
|---------|-----------|-------|
| GSS | 60 | General Social Survey — politics, religion, institutions, crime |
| OpinionQA W34 | 66 | Biomedical & food issues |
| OpinionQA W50 | 128 | Relationships & family |
| OpinionQA W92 | 77 | Political views & ideology |

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Generator model | Claude 4.5 Sonnet |
| Simulator model | Claude 4.5 Sonnet |
| Personas per source | ~1000 |
| Temperature (generation) | 0.7 |
| Temperature (simulation) | 0.0 |
| Response format | Single option selection |
| Aggregation | Vote counting |
| Inference mode | AWS Bedrock Batch |

## Data Formats

### Per-user behavioral data (input to `generate_personas.py`)

```json
[{
  "id": "user_1",
  "feature": {"age": "33", "gender": "male"},
  "responses": [
    {"Question": "...", "Possible_Answers": "A; B", "Response": "A"}
  ]
}]
```

### Benchmark survey (input to `run_inference.py`)

```json
{
  "survey_title": "Global Social Survey (GSS)",
  "questions": [{
    "question_id": "0",
    "input_template": "Do you think the use of marijuana should be made legal or not?",
    "answer_options": {"1": "Should be legal", "2": "Should not be legal"},
    "human_counts": {"1": 594, "2": 273},
    "type": "multiple_choice"
  }]
}
```

### Persona output

```json
[{
  "short_bio": "You are a politically moderate individual who...",
  "user_profile": {
    "age": {"value": "30-49", "confidence": "high"},
    "gender": {"value": "Male", "confidence": "high"}
  },
  "user_id": "socsci_mp3dr_1"
}]
```
