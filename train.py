"""Trains UFC fight-winner models and evaluates them on fights they've never seen."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import download, load_bout_stats, load_fighters, load_fights
from features import FEATURES, build_dataset

FEATURE_COLS = [f"{name}_diff" for name in FEATURES]
SPLIT_DATE = "2024-01-01"  # train on the past, test on the most recent fights
MODEL_PATH = Path(__file__).parent / "model.joblib"


def main():
    download()
    fighters = load_fighters()
    data, records = build_dataset(load_fights(), load_bout_stats(), fighters)

    # Time-based split: like real life, we predict future fights from past ones.
    train = data[data["date"] < SPLIT_DATE]
    test = data[data["date"] >= SPLIT_DATE]
    X_train, y_train = train[FEATURE_COLS], train["red_won"]
    X_test, y_test = test[FEATURE_COLS], test["red_won"]
    print(f"Training on {len(train)} fights before {SPLIT_DATE}, testing on {len(test)} after\n")

    # Baseline 1: always pick red (should be ~50% thanks to the random swap).
    print(f"Baseline - always pick red:          {(y_test == 1).mean():.3f}")
    # Baseline 2: pick whoever has more UFC wins (a simple human-style rule).
    more_wins = (X_test["wins_diff"] > 0).astype(int)
    print(f"Baseline - pick fighter with more wins: {(more_wins == y_test).mean():.3f}")

    # Model 1: logistic regression (needs missing values filled and features scaled).
    logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression())
    logreg.fit(X_train, y_train)
    print(f"Logistic regression test accuracy:   {logreg.score(X_test, y_test):.3f}")

    # Model 2: gradient boosted trees (handles missing values on its own).
    gbt = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=42)
    gbt.fit(X_train, y_train)
    print(f"Gradient boosting train accuracy:    {gbt.score(X_train, y_train):.3f}")
    print(f"Gradient boosting test accuracy:     {gbt.score(X_test, y_test):.3f}\n")

    # Which features matter? Shuffle one at a time and see how much accuracy drops.
    importance = permutation_importance(gbt, X_test, y_test, n_repeats=5, random_state=42)
    ranked = pd.Series(importance.importances_mean, index=FEATURE_COLS).sort_values(ascending=False)
    print("Most useful features:")
    print(ranked.head(6).round(4).to_string(), "\n")

    # Keep whichever model did better on the test fights, retrain it on ALL fights
    # for real predictions, and save what predict.py needs.
    best = logreg if logreg.score(X_test, y_test) >= gbt.score(X_test, y_test) else gbt
    best.fit(data[FEATURE_COLS], data["red_won"])
    joblib.dump({"model": best, "records": records, "fighters": fighters}, MODEL_PATH)
    name = "logistic regression" if best is logreg else "gradient boosting"
    print(f"Saved {name} model to {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
