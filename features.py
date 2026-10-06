"""Builds pre-fight features by replaying UFC history in date order.

Key rule: every feature for a fight uses ONLY information from before that fight.
Otherwise the model would "see the future" (data leakage) and look falsely accurate.
"""

import numpy as np
import pandas as pd

FEATURES = [
    "wins", "losses", "fights", "win_rate", "streak", "days_since_last",
    "avg_sig_landed", "avg_sig_absorbed", "avg_td_landed",
    "height", "reach", "age",
]


def new_record():
    return {"wins": 0, "losses": 0, "streak": 0, "last_date": None,
            "sig_landed": 0, "sig_absorbed": 0, "td_landed": 0, "stat_fights": 0}


def fighter_snapshot(record, physical, date):
    """A fighter's numbers going INTO a fight on `date`."""
    fights = record["wins"] + record["losses"]
    n = record["stat_fights"]
    dob = physical.get("dob")
    return {
        "wins": record["wins"],
        "losses": record["losses"],
        "fights": fights,
        "win_rate": record["wins"] / fights if fights else np.nan,
        "streak": record["streak"],  # +3 = won last 3, -2 = lost last 2
        "days_since_last": (date - record["last_date"]).days if record["last_date"] else np.nan,
        "avg_sig_landed": record["sig_landed"] / n if n else np.nan,
        "avg_sig_absorbed": record["sig_absorbed"] / n if n else np.nan,
        "avg_td_landed": record["td_landed"] / n if n else np.nan,
        "height": physical.get("height", np.nan),
        "reach": physical.get("reach", np.nan),
        "age": (date - dob).days / 365.25 if pd.notna(dob) else np.nan,
    }


def matchup_features(snap_a, snap_b):
    """Model input = differences (fighter A minus fighter B), e.g. reach advantage."""
    return {f"{name}_diff": snap_a[name] - snap_b[name] for name in FEATURES}


def update_record(record, won, date, landed, absorbed, td):
    """After a fight, add its result and stats to the fighter's record."""
    if won:
        record["wins"] += 1
        record["streak"] = record["streak"] + 1 if record["streak"] > 0 else 1
    else:
        record["losses"] += 1
        record["streak"] = record["streak"] - 1 if record["streak"] < 0 else -1
    record["last_date"] = date
    if landed is not None:
        record["sig_landed"] += landed
        record["sig_absorbed"] += absorbed
        record["td_landed"] += td
        record["stat_fights"] += 1


def build_dataset(fights, bout_stats, fighters, seed=42):
    """Return (dataset with one row per fight, final records for every fighter).

    ufcstats lists the WINNER first in most fights, so we randomly swap the two
    fighters. Without this, the model could just learn "fighter A usually wins".
    """
    rng = np.random.default_rng(seed)
    records = {}
    rows = []

    for fight in fights.sort_values("date").itertuples(index=False):
        a, b, a_won, date = fight.fighter_a, fight.fighter_b, int(fight.a_won), fight.date
        rec_a = records.setdefault(a, new_record())
        rec_b = records.setdefault(b, new_record())

        # 1) Features from BEFORE the fight.
        snap_a = fighter_snapshot(rec_a, fighters.get(a, {}), date)
        snap_b = fighter_snapshot(rec_b, fighters.get(b, {}), date)
        if rng.random() < 0.5:
            row = matchup_features(snap_a, snap_b)
            row.update(red=a, blue=b, red_won=a_won)
        else:
            row = matchup_features(snap_b, snap_a)
            row.update(red=b, blue=a, red_won=1 - a_won)
        row.update(date=date, weight_class=fight.WEIGHTCLASS)
        rows.append(row)

        # 2) THEN update both records with what happened.
        key_a, key_b = (fight.EVENT, fight.BOUT, a), (fight.EVENT, fight.BOUT, b)
        if key_a in bout_stats.index and key_b in bout_stats.index:
            sa, sb = bout_stats.loc[key_a], bout_stats.loc[key_b]
            update_record(rec_a, a_won, date, sa.sig_landed, sb.sig_landed, sa.td_landed)
            update_record(rec_b, not a_won, date, sb.sig_landed, sa.sig_landed, sb.td_landed)
        else:
            update_record(rec_a, a_won, date, None, None, None)
            update_record(rec_b, not a_won, date, None, None, None)

    return pd.DataFrame(rows), records
