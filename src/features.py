"""Compute leakage-free pre-match features for every odds-matched match.

Elo, form, H2H and fatigue are computed chronologically over the FULL match history
(1990-2026), but feature rows are emitted only for the 2010-2025 matches that have
bookmaker odds (data/processed/matches_odds.csv).

Output: data/processed/dataset.csv — one row per match, symmetric player encoding
(player 1 / player 2 assigned pseudo-randomly, label y = 1 if player 1 won).
"""

import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from build_dataset import load_matches

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MERGED = os.path.join(ROOT, "data", "processed", "matches_odds.csv")
OUT = os.path.join(ROOT, "data", "processed", "dataset.csv")

ELO_K = 32.0
ELO_INIT = 1500.0
FORM_N = 25


class PlayerState:
    __slots__ = ("elo", "surf_elo", "recent", "dates", "n_matches")

    def __init__(self):
        self.elo = ELO_INIT
        self.surf_elo = defaultdict(lambda: ELO_INIT)
        self.recent = deque(maxlen=FORM_N)   # 1 = win, 0 = loss
        self.dates = deque(maxlen=60)        # match dates (fatigue)
        self.n_matches = 0


def elo_expect(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def main():
    matches = load_matches()  # same filtering/order as the merge step -> same match_id
    matches = matches.sort_values(["date", "tourney_id"], kind="stable").reset_index(drop=True)

    merged = pd.read_csv(MERGED)
    odds_ids = set(merged["match_id"])
    odds_by_id = merged.set_index("match_id")

    players: dict = defaultdict(PlayerState)
    h2h: dict = defaultdict(int)  # (a, b) -> wins of a vs b
    rng = np.random.RandomState(42)
    rows = []

    for match in matches.itertuples():
        w_id, l_id = match.winner_id, match.loser_id
        surface = match.surface if isinstance(match.surface, str) else "Hard"
        pw, pl = players[w_id], players[l_id]

        if match.match_id in odds_ids:
            od = odds_by_id.loc[match.match_id]
            # symmetric encoding: player 1 is the *winner* with probability 0.5
            p1_is_winner = bool(rng.randint(2))
            a, b = (pw, pl) if p1_is_winner else (pl, pw)
            a_id, b_id = (w_id, l_id) if p1_is_winner else (l_id, w_id)
            a_rank = match.winner_rank if p1_is_winner else match.loser_rank
            b_rank = match.loser_rank if p1_is_winner else match.winner_rank
            a_pts = match.winner_rank_points if p1_is_winner else match.loser_rank_points
            b_pts = match.loser_rank_points if p1_is_winner else match.winner_rank_points
            a_age = match.winner_age if p1_is_winner else match.loser_age
            b_age = match.loser_age if p1_is_winner else match.winner_age

            d = pd.Timestamp(match.date)
            fatigue_a = sum(1 for t in a.dates if (d - t).days <= 14)
            fatigue_b = sum(1 for t in b.dates if (d - t).days <= 14)

            def form(p, n):
                r = list(p.recent)[-n:]
                return np.mean(r) if r else np.nan

            row = {
                "match_id": match.match_id,
                "date": match.date,
                "season": od["season"],
                "surface": surface,
                "tourney_level": match.tourney_level,
                "best_of": match.best_of,
                "round": match.round,
                "y": int(p1_is_winner),
                # pre-match state differences (player1 - player2)
                "elo_diff": a.elo - b.elo,
                "surf_elo_diff": a.surf_elo[surface] - b.surf_elo[surface],
                "rank_diff_log": (np.log(a_rank) - np.log(b_rank))
                if a_rank == a_rank and b_rank == b_rank and a_rank > 0 and b_rank > 0
                else np.nan,
                "points_diff_log": (np.log1p(a_pts) - np.log1p(b_pts))
                if a_pts == a_pts and b_pts == b_pts
                else np.nan,
                "form10_diff": form(a, 10) - form(b, 10)
                if form(a, 10) == form(a, 10) and form(b, 10) == form(b, 10) else np.nan,
                "form25_diff": form(a, 25) - form(b, 25)
                if form(a, 25) == form(a, 25) and form(b, 25) == form(b, 25) else np.nan,
                "h2h_diff": h2h[(a_id, b_id)] - h2h[(b_id, a_id)],
                "fatigue14_diff": fatigue_a - fatigue_b,
                "age_diff": (a_age - b_age) if a_age == a_age and b_age == b_age else np.nan,
                "exp_diff_log": np.log1p(a.n_matches) - np.log1p(b.n_matches),
                # market odds, mapped to player1/player2
                "p1_odds_avg": od["AvgW"] if p1_is_winner else od["AvgL"],
                "p2_odds_avg": od["AvgL"] if p1_is_winner else od["AvgW"],
                "p1_odds_ps": od["PSW"] if p1_is_winner else od["PSL"],
                "p2_odds_ps": od["PSL"] if p1_is_winner else od["PSW"],
                "p1_odds_b365": od["B365W"] if p1_is_winner else od["B365L"],
                "p2_odds_b365": od["B365L"] if p1_is_winner else od["B365W"],
                "comment": od["Comment"],
            }
            rows.append(row)

        # ---- post-match updates (never used for the current row) --------------------
        exp_w = elo_expect(pw.elo, pl.elo)
        pw.elo += ELO_K * (1 - exp_w)
        pl.elo += ELO_K * (0 - (1 - exp_w))
        exp_ws = elo_expect(pw.surf_elo[surface], pl.surf_elo[surface])
        pw.surf_elo[surface] += ELO_K * (1 - exp_ws)
        pl.surf_elo[surface] += ELO_K * (0 - (1 - exp_ws))
        pw.recent.append(1)
        pl.recent.append(0)
        d = pd.Timestamp(match.date)
        pw.dates.append(d)
        pl.dates.append(d)
        pw.n_matches += 1
        pl.n_matches += 1
        h2h[(w_id, l_id)] += 1

    df = pd.DataFrame(rows)
    # drop retirements/walkovers: pre-match odds exist but the label is unreliable
    df = df[df["comment"].isin(["Completed", "Sched"]) | df["comment"].isna()]
    df.to_csv(OUT, index=False)
    print(f"Rows: {len(df)}  ->  {OUT}")
    print("NaN per feature (%):")
    print((df.isna().mean() * 100).round(1).to_string())


if __name__ == "__main__":
    main()
