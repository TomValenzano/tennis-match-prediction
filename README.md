# Tennis Match Prediction — ML Course Project

Predicting ATP match outcomes with calibrated ML models, compared against Elo and bookmaker baselines.
Machine Learning course (a.y. 2025/26) — Tommaso Valenzano, 860149.

## Project structure

```
tennis-prediction/
├── data/
│   ├── raw/
│   │   ├── matches/        # per-season ATP results, Sackmann-format CSVs (1990-2026)
│   │   └── odds/           # tennis-data.co.uk yearly files with bookmaker odds (TO ADD)
│   └── processed/          # merged dataset with engineered features
├── src/                    # pipeline scripts (elo, features, models, evaluation)
├── notebooks/              # exploratory analyses
└── results/                # metrics tables and figures
```

## Data sources

- **Match results**: [TML-Database](https://github.com/Tennismylife/TML-Database), a live-updated ATP
  match database in the same schema as Jeff Sackmann's well-known `tennis_atp` dataset (the original
  repository is no longer available on GitHub). One CSV per season; here: 1990-2026, ~115k matches,
  50 columns (players, ranking/points, surface, tournament level, per-match serve stats).
- **Bookmaker odds**: [tennis-data.co.uk](http://www.tennis-data.co.uk/alldata.php) — one file per
  season (2010-2025) with closing odds from multiple bookmakers. Download manually and place the
  yearly `.xls/.xlsx` files in `data/raw/odds/`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data (not included in this repository)

The datasets are not redistributed here for licensing reasons. To reproduce:

1. **Match results**: download the per-season CSVs (1990-2026) from
   [TML-Database](https://github.com/Tennismylife/TML-Database) into `data/raw/matches/`.
2. **Bookmaker odds**: download the yearly ATP files (2010-2025) from
   [tennis-data.co.uk](http://www.tennis-data.co.uk/alldata.php) into `data/raw/odds/`.

## Reproducing the results

```bash
cd src
python build_dataset.py     # merge results and odds (98.7% match rate)
python features.py          # chronological Elo + leakage-free features
python train.py             # temporal protocol: train <=2022, tune 2023, test 2024-25
python evaluate.py calib    # calibration (reliability diagram + ECE)
python evaluate.py boot     # paired bootstrap on the test seasons
python evaluate.py abl      # feature-group ablation
python evaluate.py bet      # flat-stake betting simulation
python evaluate.py roll     # rolling-origin splits (rerun until all seasons done),
                            # then corrected resampled t-tests
```

All seeds are fixed: a full rerun reproduces every number in the writeup exactly.

## Pipeline

1. `src/build_dataset.py` — cleans matches, excludes Davis Cup/retirements, merges with odds
2. `src/features.py` — chronological Elo (overall + per surface) and leakage-free pre-match features, symmetric encoding
3. `src/train.py` — LR / RF / gradient boosting / Keras NN with temporal validation, Elo and market baselines
4. `src/evaluate.py` — calibration, paired bootstrap, corrected resampled t-tests, ablation, betting simulation
