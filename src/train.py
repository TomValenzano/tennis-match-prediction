"""Train and evaluate the models with a temporal protocol.

Split:  train <= 2022   |   validation = 2023 (hyperparameter tuning)   |   test = 2024-25

Models: logistic regression, random forest, gradient boosting (hist), feed-forward NN.
The NN uses Keras if tensorflow is installed, otherwise sklearn's MLPClassifier
(same architecture) as a drop-in fallback.

Baselines: pure Elo (probability from the Elo logistic curve) and the bookmaker
average / Pinnacle implied probabilities (margin removed).

Outputs:
  results/metrics_test.csv        - metric table on the test seasons
  data/processed/preds_test.csv   - per-match test probabilities of every model
                                    (used later for calibration + significance analysis)
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "processed", "dataset.csv")
RESULTS = os.path.join(ROOT, "results")

NUM_FEATURES = [
    "elo_diff", "surf_elo_diff", "rank_diff_log", "points_diff_log",
    "form10_diff", "form25_diff", "h2h_diff", "fatigue14_diff",
    "age_diff", "exp_diff_log", "best_of",
]
CAT_FEATURES = ["surface", "tourney_level"]

TRAIN_END, VAL_SEASON, TEST_START = 2022, 2023, 2024
SEED = 42


def implied_prob(o1: pd.Series, o2: pd.Series) -> pd.Series:
    """Two-way margin removal: p1 = (1/o1) / (1/o1 + 1/o2)."""
    return (1 / o1) / (1 / o1 + 1 / o2)


def make_preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])


def build_models() -> dict:
    """Model name -> (needs_scaling, list of (config_name, estimator))."""
    models = {
        "logreg": (True, [
            (f"C={c}", LogisticRegression(C=c, max_iter=2000)) for c in (0.01, 0.1, 1.0)
        ]),
        "random_forest": (False, [
            (f"depth={d}", RandomForestClassifier(
                n_estimators=400, max_depth=d, min_samples_leaf=20,
                n_jobs=-1, random_state=SEED))
            for d in (8, 12, None)
        ]),
        "grad_boosting": (False, [
            (f"lr={lr},leaves={lv}", HistGradientBoostingClassifier(
                learning_rate=lr, max_leaf_nodes=lv, max_iter=400,
                early_stopping=True, validation_fraction=0.15, random_state=SEED))
            for lr, lv in ((0.05, 31), (0.1, 31), (0.05, 63))
        ]),
    }
    try:
        from scikeras.wrappers import KerasClassifier  # noqa: F401
        import tensorflow  # noqa: F401
        has_keras = True
    except Exception:
        has_keras = False
    if has_keras:
        from tensorflow import keras

        def make_nn(hidden=32, lr=1e-3):
            def build(meta):
                m = keras.Sequential([
                    keras.layers.Input(shape=(meta["n_features_in_"],)),
                    keras.layers.Dense(hidden, activation="relu"),
                    keras.layers.Dropout(0.3),
                    keras.layers.Dense(hidden // 2, activation="relu"),
                    keras.layers.Dense(1, activation="sigmoid"),
                ])
                m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy")
                return m
            from scikeras.wrappers import KerasClassifier
            return KerasClassifier(model=build, epochs=30, batch_size=256, verbose=0,
                                   validation_split=0.15,
                                   callbacks=[keras.callbacks.EarlyStopping(patience=3,
                                              restore_best_weights=True)])
        models["neural_net"] = (True, [
            ("h=32", make_nn(32)), ("h=64", make_nn(64)),
        ])
        print("NN backend: Keras")
    else:
        from sklearn.neural_network import MLPClassifier
        models["neural_net"] = (True, [
            (f"h={h}", MLPClassifier(hidden_layer_sizes=(h, h // 2), alpha=a,
                                     early_stopping=True, max_iter=300, random_state=SEED))
            for h, a in ((32, 1e-3), (64, 1e-3), (64, 1e-2))
        ])
        print("NN backend: sklearn MLP (tensorflow not installed)")
    return models


def evaluate(y, p) -> dict:
    return {
        "accuracy": accuracy_score(y, p >= 0.5),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p),
        "n": len(y),
    }


def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    train = df[df.season <= TRAIN_END]
    val = df[df.season == VAL_SEASON]
    trainval = df[df.season <= VAL_SEASON]
    test = df[df.season >= TEST_START].copy()
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    X_cols = NUM_FEATURES + CAT_FEATURES
    results, test_preds = [], {}

    for name, (scale, configs) in build_models().items():
        best, best_ll = None, np.inf
        for cfg_name, est in configs:
            pipe = Pipeline([("prep", make_preprocessor(scale)), ("model", est)])
            pipe.fit(train[X_cols], train["y"])
            p_val = pipe.predict_proba(val[X_cols])[:, 1]
            ll = log_loss(val["y"], np.clip(p_val, 1e-6, 1 - 1e-6))
            print(f"  {name} [{cfg_name}] val log-loss = {ll:.4f}")
            if ll < best_ll:
                best_ll, best = ll, (cfg_name, est)
        # retrain the selected config on train+val, evaluate on test
        cfg_name, est = best
        pipe = Pipeline([("prep", make_preprocessor(scale)), ("model", est)])
        pipe.fit(trainval[X_cols], trainval["y"])
        p_test = pipe.predict_proba(test[X_cols])[:, 1]
        test_preds[name] = p_test
        res = {"model": f"{name} ({cfg_name})"} | evaluate(test["y"].values, p_test)
        results.append(res)
        print(f"-> {name} best: {cfg_name}  (val ll {best_ll:.4f})")

    # ---- baselines on the same test set ---------------------------------------------
    p_elo = 1 / (1 + 10 ** (-(test["elo_diff"]) / 400))
    results.append({"model": "BASELINE elo"} | evaluate(test["y"].values, p_elo.values))
    test_preds["elo"] = p_elo.values

    for label, (c1, c2) in {"market_avg": ("p1_odds_avg", "p2_odds_avg"),
                            "market_pinnacle": ("p1_odds_ps", "p2_odds_ps")}.items():
        mask = test[c1].notna() & test[c2].notna()
        p_mkt = implied_prob(test.loc[mask, c1], test.loc[mask, c2])
        results.append({"model": f"BASELINE {label}"} |
                       evaluate(test.loc[mask, "y"].values, p_mkt.values))
        full = np.full(len(test), np.nan)
        full[mask.values] = p_mkt.values
        test_preds[label] = full

    out = pd.DataFrame(results)
    os.makedirs(RESULTS, exist_ok=True)
    out.to_csv(os.path.join(RESULTS, "metrics_test.csv"), index=False)
    print("\n" + out.to_string(index=False))

    preds = test[["match_id", "date", "season", "y"]].copy()
    for k, v in test_preds.items():
        preds[f"p_{k}"] = v
    preds.to_csv(os.path.join(ROOT, "data", "processed", "preds_test.csv"), index=False)
    print("\nSaved metrics and test predictions.")


if __name__ == "__main__":
    main()
