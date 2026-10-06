# UFC Fight Predictor

A machine learning model that predicts the winner of UFC fights from each fighter's record, fighting stats and physical attributes, trained on every UFC fight since 1994.

```
$ python predict.py "Islam Makhachev" "Charles Oliveira"
Islam Makhachev: 73%   Charles Oliveira: 27%
```

## Results

Trained on fights before 2024, tested on ~1,450 fights from 2024 onward that the model never saw.

| Model | Test accuracy |
|---|---|
| Always pick one corner | ~47% |
| Pick the fighter with more UFC wins | ~48% |
| **Logistic regression** | **~62%** |
| Gradient boosted trees | ~59% |

For context, betting favourites win roughly two-thirds of UFC fights, so ~62% from public stats alone is a reasonable result. The most useful features were **age difference**, **average significant strikes absorbed** and **losses**.

## How it works

| File | Job |
|---|---|
| `data.py` | Downloads ufcstats.com data and parses heights, reach, strike counts and results |
| `features.py` | Replays UFC history in date order to build each fighter's record *going into* every fight |
| `train.py` | Time-based train/test split, two baselines, two models, permutation feature importance |
| `predict.py` | Predicts any matchup using fighters' current records |
| `test_ufc.py` | Unit tests, including a test that features never use future fights |

### Design decisions
- **No data leakage:** features for each fight only use fights that happened before it.
- **Random corner swap:** the source data lists the winner first, so fighters are randomly swapped; otherwise a model could just learn "the first fighter wins".
- **Time-based split:** the model is tested on fights after its training data, like real predictions.
- **Order-independent predictions:** `predict.py` averages both corner orders.

## Run it

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
python predict.py "Fighter One" "Fighter Two"
pytest
```

Data: [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats), scraped from ufcstats.com.
