"""Merge TML match results (Sackmann schema) with tennis-data.co.uk bookmaker odds.

Output: data/processed/matches_odds.csv  (one row per matched match, 2010-2025)

Join strategy: matches are paired on (player-name keys, date window, tournament round
proximity). Names are normalised to "surname + first initial" keys on both sides;
multi-word surnames are handled by trying several splits on the results side.
"""

import glob
import os
import re
import unicodedata

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MATCHES_DIR = os.path.join(ROOT, "data", "raw", "matches")
ODDS_DIR = os.path.join(ROOT, "data", "raw", "odds")
OUT_PATH = os.path.join(ROOT, "data", "processed", "matches_odds.csv")

ODDS_YEARS = range(2010, 2026)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def norm(s: str) -> str:
    s = strip_accents(s).lower().replace("-", " ").replace("'", "")
    return re.sub(r"\s+", " ", s).strip()


def odds_name_key(name: str) -> str:
    """'Popyrin A.' -> 'popyrin|a'; 'De Minaur A.' -> 'deminaur|a'; 'Zhang Zh.' -> 'zhang|z'.

    Surnames are compared with spaces removed, so "O Connell" == "O'Connell" == "OConnell".
    """
    s = norm(name).replace(".", " ")
    s = re.sub(r"\bjr\b", "", s)
    parts = [p for p in re.sub(r"\s+", " ", s).strip().split(" ") if p]
    # trailing short tokens (1-2 chars) are initials ("A", "Zh", "J L")
    initials = []
    while len(parts) > 1 and len(parts[-1]) <= 2:
        initials.insert(0, parts.pop())
    surname = "".join(parts)
    ini = initials[0][0] if initials else ""
    return f"{surname}|{ini}"


def odds_first_surname_key(name: str) -> str:
    """First surname token only: 'Mpetshi G.' -> 'mpetshi|g' (matches 'Mpetshi Perricard')."""
    s = norm(name).replace(".", " ")
    parts = [p for p in re.sub(r"\s+", " ", s).strip().split(" ") if p]
    initials = []
    while len(parts) > 1 and len(parts[-1]) <= 2:
        initials.insert(0, parts.pop())
    ini = initials[0][0] if initials else ""
    return f"{parts[0]}|{ini}"


def result_name_keys(name: str) -> list[str]:
    """'Alex De Minaur' -> ['deminaur|a', 'minaur|a']; 'Grigor Dimitrov' -> ['dimitrov|g']."""
    parts = norm(name).split(" ")
    if len(parts) < 2:
        return [f"{parts[0]}|"]
    ini = parts[0][0]
    keys = []
    for i in range(1, len(parts)):
        keys.append(f"{''.join(parts[i:])}|{ini}")
    return keys


def result_first_surname_keys(name: str) -> list[str]:
    """Keys using each single word as surname: 'Giovanni Mpetshi Perricard' ->
    ['mpetshi|g', 'perricard|g'] (first-token fallback for compound surnames)."""
    parts = norm(name).split(" ")
    if len(parts) < 2:
        return []
    ini = parts[0][0]
    return [f"{p}|{ini}" for p in parts[1:]]


def load_matches() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(MATCHES_DIR, "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    # keep tour-level singles; drop Davis Cup / exhibitions
    df = df[df["tourney_level"].isin(["G", "M", "500", "250", "F", "O"])].copy()
    # NOTE: match_num is NaN in several seasons -> use the row position as unique id
    df = df.reset_index(drop=True)
    df["match_id"] = df.index
    return df


def load_odds() -> pd.DataFrame:
    frames = []
    for y in ODDS_YEARS:
        for ext in ("xlsx", "xls"):
            p = os.path.join(ODDS_DIR, f"{y}.{ext}")
            if os.path.exists(p):
                d = pd.read_excel(p)
                d["season"] = y
                frames.append(d)
                break
    odds = pd.concat(frames, ignore_index=True)
    odds["Date"] = pd.to_datetime(odds["Date"])
    odds["w_key"] = odds["Winner"].map(odds_name_key)
    odds["l_key"] = odds["Loser"].map(odds_name_key)
    odds["odds_id"] = np.arange(len(odds))
    return odds


def _join_and_dedup(mw: pd.DataFrame, odds: pd.DataFrame, matches_cols: list) -> pd.DataFrame:
    merged = mw.merge(
        odds,
        left_on=["w_keys", "l_keys"],
        right_on=["w_key", "l_key"],
        how="inner",
        suffixes=("", "_od"),
    )
    # tournaments span ~2 weeks; allow generous window between tourney start and odds date
    delta = (merged["Date"] - merged["date"]).dt.days
    merged = merged[(delta >= -3) & (delta <= 25)].copy()

    # deduplicate: keep the closest date, then the best rank agreement
    merged["date_gap"] = (merged["Date"] - merged["date"]).dt.days.abs()
    merged["rank_gap"] = (merged["WRank"].fillna(9999) - merged["winner_rank"].fillna(9999)).abs()
    merged = merged.sort_values(["match_id", "date_gap", "rank_gap"])
    merged = merged.drop_duplicates("match_id", keep="first")
    merged = merged.sort_values(["odds_id", "date_gap", "rank_gap"]).drop_duplicates(
        "odds_id", keep="first"
    )
    keep_cols = matches_cols + [
        "odds_id", "season", "Date", "Series", "Court", "WRank", "LRank",
        "B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL", "Comment",
    ]
    return merged[keep_cols]


def merge(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    m = matches[(matches["date"] >= "2009-12-01")].copy()
    matches_cols = list(matches.columns)

    # ---- pass 1: full-surname keys -------------------------------------------------
    m["w_keys"] = m["winner_name"].map(result_name_keys)
    m["l_keys"] = m["loser_name"].map(result_name_keys)
    mw = m.explode("w_keys").explode("l_keys")
    merged1 = _join_and_dedup(mw, odds, matches_cols)

    # ---- pass 2: first-surname-token fallback for the leftovers --------------------
    left_odds = odds[~odds["odds_id"].isin(merged1["odds_id"])].copy()
    left_m = m[~m["match_id"].isin(merged1["match_id"])].copy()
    left_odds["w_key"] = left_odds["Winner"].map(odds_first_surname_key)
    left_odds["l_key"] = left_odds["Loser"].map(odds_first_surname_key)
    left_m["w_keys"] = left_m["winner_name"].map(
        lambda s: result_name_keys(s) + result_first_surname_keys(s)
    )
    left_m["l_keys"] = left_m["loser_name"].map(
        lambda s: result_name_keys(s) + result_first_surname_keys(s)
    )
    mw2 = left_m.explode("w_keys").explode("l_keys")
    merged2 = _join_and_dedup(mw2, left_odds, matches_cols)
    # first-token keys are ambiguous -> require rank agreement as a safety check
    ranks_ok = (
        (merged2["WRank"].isna())
        | (merged2["winner_rank"].isna())
        | ((merged2["WRank"] - merged2["winner_rank"]).abs() <= 5)
    )
    merged2 = merged2[ranks_ok]

    merged = pd.concat([merged1, merged2], ignore_index=True)
    merged = merged.drop_duplicates("odds_id").drop_duplicates("match_id")
    return merged.drop(columns=["odds_id", "WRank", "LRank"])


def main():
    matches = load_matches()
    odds = load_odds()
    merged = merge(matches, odds)

    n_odds = len(odds)
    print(f"Odds rows 2010-2025:      {n_odds}")
    print(f"Matched to results:       {len(merged)}  ({len(merged)/n_odds:.1%})")
    completed = odds  # informational only
    per_season = merged.groupby("season").size()
    print(per_season.to_string())

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
