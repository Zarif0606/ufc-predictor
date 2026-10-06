"""Downloads and cleans UFC data scraped from ufcstats.com.

Data source: github.com/Greco1899/scrape_ufc_stats (updated after every event).
"""

import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
BASE_URL = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"
FILES = ["ufc_event_details.csv", "ufc_fight_results.csv", "ufc_fight_stats.csv", "ufc_fighter_tott.csv"]


def download(force=False):
    """Download the CSVs into data/ (use force=True to get the newest events)."""
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        path = DATA_DIR / name
        if force or not path.exists():
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(BASE_URL + name, path)


# ---------- small parsing helpers (unit tested) ----------

def parse_inches(text):
    """'5' 11"' -> 71.0, '66"' -> 66.0, '--' -> None."""
    text = str(text).strip()
    if text in ("--", "", "nan"):
        return None
    if "'" in text:
        feet, inches = text.replace('"', "").split("'")
        return int(feet) * 12 + int(inches.strip() or 0)
    return float(text.replace('"', ""))


def parse_landed(text):
    """'16 of 40' -> 16 (strikes or takedowns landed)."""
    try:
        return int(str(text).split(" of ")[0])
    except ValueError:
        return 0


def parse_outcome(outcome):
    """'W/L' -> 1 (first fighter won), 'L/W' -> 0, draws / no contests -> None."""
    return {"W/L": 1, "L/W": 0}.get(str(outcome).strip())


# ---------- loading ----------

def load_fights():
    """One row per decided fight: date, fighter_a, fighter_b, a_won, weight class."""
    events = pd.read_csv(DATA_DIR / "ufc_event_details.csv")
    events["EVENT"] = events["EVENT"].str.strip()
    events["date"] = pd.to_datetime(events["DATE"], format="%B %d, %Y")

    results = pd.read_csv(DATA_DIR / "ufc_fight_results.csv")
    results["EVENT"] = results["EVENT"].str.strip()
    results["BOUT"] = results["BOUT"].str.strip()
    results["a_won"] = results["OUTCOME"].map(parse_outcome)
    results = results.dropna(subset=["a_won"])  # drop draws and no contests

    fighters = results["BOUT"].str.split(" vs. ", n=1, expand=True)
    results["fighter_a"] = fighters[0].str.strip()
    results["fighter_b"] = fighters[1].str.strip()

    fights = results.merge(events[["EVENT", "date"]], on="EVENT")
    fights = fights.dropna(subset=["fighter_a", "fighter_b"])
    return fights[["date", "EVENT", "BOUT", "fighter_a", "fighter_b", "a_won", "WEIGHTCLASS"]]


def load_bout_stats():
    """Total significant strikes and takedowns landed per fighter per bout."""
    stats = pd.read_csv(DATA_DIR / "ufc_fight_stats.csv")
    stats = stats.dropna(subset=["FIGHTER"])
    stats["sig_landed"] = stats["SIG.STR."].map(parse_landed)
    stats["td_landed"] = stats["TD"].map(parse_landed)
    for col in ["EVENT", "BOUT", "FIGHTER"]:
        stats[col] = stats[col].str.strip()
    return stats.groupby(["EVENT", "BOUT", "FIGHTER"])[["sig_landed", "td_landed"]].sum()


def load_fighters():
    """Height, reach (inches) and date of birth per fighter name."""
    tott = pd.read_csv(DATA_DIR / "ufc_fighter_tott.csv")
    tott["FIGHTER"] = tott["FIGHTER"].str.strip()
    tott = tott.drop_duplicates("FIGHTER", keep=False)  # skip names shared by 2+ fighters
    tott["height"] = tott["HEIGHT"].map(parse_inches)
    tott["reach"] = tott["REACH"].map(parse_inches)
    tott["dob"] = pd.to_datetime(tott["DOB"], format="%b %d, %Y", errors="coerce")
    return tott.set_index("FIGHTER")[["height", "reach", "dob"]].to_dict("index")
