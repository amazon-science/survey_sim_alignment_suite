# Data

No data files are included in this repository. Follow the steps below to obtain them.

## Benchmarks (`data/benchmarks/`)

Survey benchmark files containing questions and human response distributions.

**GSS**: 60 multiple-choice questions from the General Social Survey (https://gss.norc.org/get-the-data/stata), using the following variables:

```
GRASS, CAPPUN, ABDEFECT, ABNOMORE, ABHLTH, ABPOOR, ABRAPE, ABSINGLE, ABANY,
POSTLIFE, SPKATH, COLATH, LIBATH, TRUST, FAIR, HELPFUL, VOTE20, VOTE16, FEAR,
OWNGUN, GUNLAW, LIFE, CONFINAN, CONBUS, CONCLERG, CONEDUC, CONFED, CONJUDGE,
CONLABOR, COURTS, NATSPAC, NATMASS, NATPRESS, NATARMS, NATEDUC, NATHEAL,
NATDRUG, NATENVIR, NATAID, NATCITY, BIBLE, NATPARK, FINALTER, FINRELA, HAPPY,
HAPMAR, NATRACE, WELFARE, SATJOB, PRES16, XMARSEX, SEXEDUC, HOMOSEX, CLASS,
HEALTH, PRES20, PRAY, POLVIEWS, PARTYID, ATTEND
```

For each variable, aggregate individual responses into counts per option, excluding missing/refused. Format as JSON (see below).

**OpinionQA**: Download from https://worksheets.codalab.org/worksheets/0x6fb693719477478aac73fc07db333f69 (original dataset from Santurkar et al., 2023). Select waves 34, 50, and 92. For each question, extract the question text, response options, and aggregate response counts across all respondents. Format as JSON (see below).

Expected output format:
```json
{
  "survey_title": "...",
  "benchmark_id": "gss",
  "survey_type": "opinion",
  "questions": [{
    "question_id": "0",
    "input_template": "Do you think the use of marijuana should be made legal or not?",
    "answer_options": {"1": "Should be legal", "2": "Should not be legal"},
    "human_counts": {"1": 594, "2": 273},
    "type": "multiple_choice"
  }]
}
```

Expected files:
- `gss.json` — 60 questions
- `opinionqa_w34.json` — 66 questions
- `opinionqa_w50.json` — 128 questions
- `opinionqa_w92.json` — 77 questions

## Raw data (`data/raw/`)

Download manually:

| Source | URL | Files needed | Destination |
|--------|-----|--------------|-------------|
| SocSci210 | https://huggingface.co/datasets/socratesft/SocSci210 | Auto-downloaded by `prepare_socsci210.py` | N/A |
| Rotten Tomatoes | https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset | `rotten_tomatoes_critic_reviews.csv`, `rotten_tomatoes_movies.csv` | `data/raw/rotten_tomatoes/` |
| E-commerce | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YGLYDY | `amazon-purchases.csv`, `survey.csv` | `data/raw/ecommerce/` |

## Sample data (`data/sample/`)

Per-user behavioral records for persona generation. Regenerate after downloading raw data:

```bash
python scripts/prepare_socsci210.py --study_id mp3dr --max_users 1000
python scripts/prepare_rotten_tomatoes.py --data_dir data/raw/rotten_tomatoes/ --max_users 1000
python scripts/prepare_ecommerce.py --data_dir data/raw/ecommerce/ --max_users 1000
```
