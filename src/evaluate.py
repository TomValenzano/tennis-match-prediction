"""Deep-dive evaluation on the test seasons (2024-25).

1. Calibration: reliability diagram + expected calibration error (models vs market).
2. Paired bootstrap CIs on metric differences between predictors.
3. Corrected resampled t-test over rolling-origin temporal splits (Nadeau-Bengio).
4. Feature-group ablation (logistic regression).
5. Illustrative flat-stake betting simulation against average market odds.

Outputs in results/: calibration.png, ece.csv, bootstrap_tests.csv,
rolling_significance.csv, ablation.csv, betting.csv
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from train import (NUM_FEATURES, CAT_FEATURES, make_preprocessor, build_models,
                   implied_prob, evaluate as metric_dict)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "processed", "dataset.csv")
PREDS = os.path.join(ROOT, "data", "processed", "preds_test.csv")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
SEED = 42

MODELS = ["logreg", "random_forest", "grad_boosting", "neural_net"]
X_COLS = NUM_FEATURES + CAT_FEATURES


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    total, err = len(y), 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            err += m.sum() / total * abs(y[m].mean() - p[m].mean())
    return err


def clip(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


# ---------------------------------------------------------------- 1. calibration ----
def calibration(preds):
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    rows = []
    for name, col, style in [
        ("Logistic regression", "p_logreg", "o-"),
        ("Neural net", "p_neural_net", "s-"),
        ("Gradient boosting", "p_grad_boosting", "^-"),
        ("Elo", "p_elo", "v-"),
        ("Market (avg)", "p_market_avg", "d-"),
    ]:
        m = preds[col].notna()
        y, p = preds.loc[m, "y"].values, preds.loc[m, col].values
        edges = np.linspace(0, 1, 11)
        idx = np.digitize(p, edges[1:-1])
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() > 30]
        ys = [y[idx == b].mean() for b in range(10) if (idx == b).sum() > 30]
        plt.plot(xs, ys, style, ms=4, lw=1.2, label=name)
        rows.append({"predictor": name, "ECE": ece(y, p), "n": m.sum()})
    plt.xlabel("Predicted probability (player 1 wins)")
    plt.ylabel("Empirical frequency")
    plt.title("Reliability diagram — test seasons 2024-25")
    plt.legend()
    os.makedirs(FIG, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "calibration.png"), dpi=150)
    plt.close()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "ece.csv"), index=False)
    return df


# ------------------------------------------------------- 2. paired bootstrap CIs ----
def paired_bootstrap(preds, pairs, B=2000):
    rng = np.random.RandomState(SEED)
    rows = []
    for a, b in pairs:
        m = preds[f"p_{a}"].notna() & preds[f"p_{b}"].notna()
        y = preds.loc[m, "y"].values
        pa, pb = clip(preds.loc[m, f"p_{a}"].values), clip(preds.loc[m, f"p_{b}"].values)
        n = len(y)
        d_ll, d_br, d_acc = [], [], []
        for _ in range(B):
            i = rng.randint(0, n, n)
            d_ll.append(log_loss(y[i], pa[i]) - log_loss(y[i], pb[i]))
            d_br.append(brier_score_loss(y[i], pa[i]) - brier_score_loss(y[i], pb[i]))
            d_acc.append(accuracy_score(y[i], pa[i] >= .5) - accuracy_score(y[i], pb[i] >= .5))
        for metric, d in [("log_loss", d_ll), ("brier", d_br), ("accuracy", d_acc)]:
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append({"pair": f"{a} - {b}", "metric": metric,
                         "diff_mean": np.mean(d), "ci_lo": lo, "ci_hi": hi,
                         "significant": bool(lo > 0 or hi < 0)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "bootstrap_tests.csv"), index=False)
    return df


# ----------------------------------- 3. corrected t-test over rolling-origin splits --
def rolling_significance(df_all, first_test=2016, last_test=2025):
    """Retrain each model on all seasons < y and test on season y, for y in range.
    Then Nadeau-Bengio corrected paired t-test on per-season log-losses."""
    fixed = {
        "logreg": (True, LogisticRegression(C=1.0, max_iter=2000)),
    }
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    fixed["random_forest"] = (False, RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=20, n_jobs=-1, random_state=SEED))
    fixed["grad_boosting"] = (False, HistGradientBoostingClassifier(
        learning_rate=0.05, max_leaf_nodes=31, max_iter=400, early_stopping=True,
        validation_fraction=0.15, random_state=SEED))
    fixed["neural_net"] = (True, MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2,
                           early_stopping=True, max_iter=300, random_state=SEED))

    per_season_path = os.path.join(RES, "rolling_logloss_per_season.csv")
    done = {}
    if os.path.exists(per_season_path):
        done = pd.read_csv(per_season_path, index_col=0).to_dict("index")

    seasons = list(range(first_test, last_test + 1))
    for y in seasons:
        if y in done:
            continue
        tr = df_all[df_all.season < y]
        te = df_all[df_all.season == y]
        row = {"ratio": len(te) / len(tr)}
        for name, (scale, est) in fixed.items():
            from sklearn.base import clone
            pipe = Pipeline([("prep", make_preprocessor(scale)), ("model", clone(est))])
            pipe.fit(tr[X_COLS], tr["y"])
            p = clip(pipe.predict_proba(te[X_COLS])[:, 1])
            row[name] = log_loss(te["y"], p)
        row["elo"] = log_loss(te["y"], clip(1 / (1 + 10 ** (-te["elo_diff"] / 400))))
        m = te["p1_odds_avg"].notna() & te["p2_odds_avg"].notna()
        row["market_avg"] = log_loss(
            te.loc[m, "y"], clip(implied_prob(te.loc[m, "p1_odds_avg"],
                                              te.loc[m, "p2_odds_avg"])))
        done[y] = row
        pd.DataFrame.from_dict(done, orient="index").to_csv(per_season_path)
        print(f"  season {y} done")
        return None  # one season per invocation, to keep runs short and resumable; rerun to continue

    # all seasons computed -> corrected paired t-tests
    tab = pd.DataFrame.from_dict(done, orient="index")
    k = len(tab)
    rho = float(tab["ratio"].mean())  # n_test/n_train correction
    ll = {c: tab[c].values for c in tab.columns if c != "ratio"}
    rows = []
    names = list(ll)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = np.array(ll[names[i]]) - np.array(ll[names[j]])
            var = d.var(ddof=1)
            t = d.mean() / np.sqrt((1 / k + rho) * var) if var > 0 else np.inf
            p = 2 * (1 - stats.t.cdf(abs(t), df=k - 1))
            rows.append({"pair": f"{names[i]} - {names[j]}", "mean_diff_ll": d.mean(),
                         "t_corrected": t, "p_value": p, "k": k,
                         "significant_5pct": bool(p < 0.05)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "rolling_significance.csv"), index=False)
    return out


# ------------------------------------------------------------------ 4. ablation -----
GROUPS = {
    "elo": ["elo_diff", "surf_elo_diff"],
    "ranking": ["rank_diff_log", "points_diff_log"],
    "form": ["form10_diff", "form25_diff"],
    "h2h": ["h2h_diff"],
    "fatigue": ["fatigue14_diff"],
    "demographics": ["age_diff", "exp_diff_log"],
}


def ablation(df_all):
    trainval = df_all[df_all.season <= 2023]
    test = df_all[df_all.season >= 2024]
    rows = []

    def fit_eval(num_feats, label):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        prep = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ])
        pipe = Pipeline([("prep", prep),
                         ("model", LogisticRegression(C=1.0, max_iter=2000))])
        cols = num_feats + CAT_FEATURES
        pipe.fit(trainval[cols], trainval["y"])
        p = clip(pipe.predict_proba(test[cols])[:, 1])
        rows.append({"config": label,
                     "accuracy": accuracy_score(test["y"], p >= .5),
                     "log_loss": log_loss(test["y"], p)})

    fit_eval(NUM_FEATURES, "full")
    for g, feats in GROUPS.items():
        fit_eval([f for f in NUM_FEATURES if f not in feats], f"without {g}")
        fit_eval([f for f in feats], f"only {g}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "ablation.csv"), index=False)
    return out


# ------------------------------------------------------------- 5. betting sim -------
def betting(preds, dataset, threshold=0.05):
    d = preds.merge(dataset[["match_id", "p1_odds_avg", "p2_odds_avg"]], on="match_id")
    rows = []
    for mdl in MODELS + ["elo"]:
        p1 = d[f"p_{mdl}"]
        m = p1.notna() & d.p1_odds_avg.notna() & d.p2_odds_avg.notna()
        dd = d[m]
        stake = ret = nbets = 0
        for _, r in dd.iterrows():
            for prob, odds, win in [(r[f"p_{mdl}"], r.p1_odds_avg, r.y == 1),
                                    (1 - r[f"p_{mdl}"], r.p2_odds_avg, r.y == 0)]:
                if prob * odds > 1 + threshold:
                    stake += 1
                    nbets += 1
                    ret += odds if win else 0
        roi = (ret - stake) / stake if stake else np.nan
        rows.append({"model": mdl, "bets": nbets, "roi": roi})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "betting.csv"), index=False)
    return out


def main(step: str):
    preds = pd.read_csv(PREDS)
    dataset = pd.read_csv(DATA)

    if step == "calib":
        print("== Calibration (ECE) ==")
        print(calibration(preds).to_string(index=False))
    elif step == "boot":
        print("== Paired bootstrap on test 2024-25 ==")
        pairs = ([(m, "market_avg") for m in MODELS] + [(m, "elo") for m in MODELS]
                 + [("neural_net", "logreg"), ("grad_boosting", "logreg")])
        print(paired_bootstrap(preds, pairs, B=1000).to_string(index=False))
    elif step == "abl":
        print("== Ablation (logistic regression) ==")
        print(ablation(dataset).to_string(index=False))
    elif step == "bet":
        print("== Betting simulation (threshold 5%) ==")
        print(betting(preds, dataset).to_string(index=False))
    elif step == "roll":
        out = rolling_significance(dataset)
        if out is not None:
            print("== Rolling-origin corrected t-test (2016-2025) ==")
            print(out.to_string(index=False))
        else:
            print("(partial - rerun 'roll' until all seasons are computed)")
    else:
        raise SystemExit("usage: evaluate.py [calib|boot|abl|bet|roll]")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "calib")
