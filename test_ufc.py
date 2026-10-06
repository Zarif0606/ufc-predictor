import pandas as pd

from data import parse_inches, parse_landed, parse_outcome
from features import build_dataset


def test_parse_inches():
    assert parse_inches("5' 11\"") == 71
    assert parse_inches('66"') == 66
    assert parse_inches("--") is None


def test_parse_landed():
    assert parse_landed("16 of 40") == 16
    assert parse_landed("---") == 0


def test_parse_outcome():
    assert parse_outcome("W/L") == 1
    assert parse_outcome("L/W") == 0
    assert parse_outcome("D/D") is None


def test_features_only_use_past_fights():
    fights = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-06-01"]),
        "EVENT": ["E1", "E2"], "BOUT": ["A vs. B", "A vs. C"],
        "fighter_a": ["A", "A"], "fighter_b": ["B", "C"],
        "a_won": [1, 1], "WEIGHTCLASS": ["Lightweight", "Lightweight"],
    })
    empty_stats = pd.DataFrame(columns=["sig_landed", "td_landed"]).set_index(
        pd.MultiIndex.from_tuples([], names=["EVENT", "BOUT", "FIGHTER"]))
    data, records = build_dataset(fights, empty_stats, {})

    # First fight: nobody has any wins yet, so the wins difference must be 0.
    assert data.iloc[0]["wins_diff"] == 0
    # Second fight: A has 1 win going in, C has 0 -> difference is +1 or -1 (corner swap).
    assert abs(data.iloc[1]["wins_diff"]) == 1
    # After both fights, A's record is 2-0.
    assert records["A"]["wins"] == 2 and records["A"]["streak"] == 2
