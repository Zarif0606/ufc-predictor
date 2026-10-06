"""Predict a UFC fight:  python predict.py "Fighter One" "Fighter Two" """

import sys
from pathlib import Path

import joblib
import pandas as pd

from features import fighter_snapshot, matchup_features, new_record
from train import FEATURE_COLS

saved = joblib.load(Path(__file__).parent / "model.joblib")
model, records, fighters = saved["model"], saved["records"], saved["fighters"]


def win_probability(a, b):
    """Chance that `a` beats `b`, averaged over both corner orders so order doesn't matter."""
    today = pd.Timestamp.today().normalize()
    snap_a = fighter_snapshot(records.get(a, new_record()), fighters.get(a, {}), today)
    snap_b = fighter_snapshot(records.get(b, new_record()), fighters.get(b, {}), today)
    a_first = pd.DataFrame([matchup_features(snap_a, snap_b)])[FEATURE_COLS]
    b_first = pd.DataFrame([matchup_features(snap_b, snap_a)])[FEATURE_COLS]
    p1 = model.predict_proba(a_first)[0][1]
    p2 = 1 - model.predict_proba(b_first)[0][1]
    return (p1 + p2) / 2


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit('Usage: python predict.py "Fighter One" "Fighter Two"')
    a, b = sys.argv[1], sys.argv[2]
    for name in (a, b):
        if name not in records:
            print(f"Warning: no UFC fights found for '{name}' (check spelling)")
    p = win_probability(a, b)
    print(f"{a}: {p:.0%}   {b}: {1 - p:.0%}")
