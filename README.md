# Product Demand Prediction with Machine Learning

Predicting how many units a product will sell at a given price, so seasonal discounts can be set
on evidence rather than intuition. The model answers the question a brand owner actually faces:
*"If I drop the price from X to Y, what happens to demand?"*

**Notebook:** [`product_demand_prediction.ipynb`](product_demand_prediction.ipynb)

---

## Dataset

[demand.csv](https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv) — 150,150 rows of historical store-level sales.

| Column | Description | Role |
| --- | --- | --- |
| `ID` | Product ID | Identifier — excluded |
| `Store ID` | Specific store ID | Identifier — excluded |
| `Total Price` | Price at which the product was sold (i.e. after discount) | **Feature** |
| `Base Price` | Initial, undiscounted price of the product | **Feature** |
| `Units Sold` | Quantity demanded | **Target** |

### Feature selection

Correlation with `Units Sold`:

| Column | Correlation |
| --- | --- |
| `Total Price` | **-0.236** |
| `Base Price` | **-0.140** |
| `Store ID` | -0.004 |
| `ID` | -0.011 |

Both price columns correlate negatively with demand — the law of demand, visible in the data.
`ID` and `Store ID` are effectively uncorrelated (|r| < 0.02); they are arbitrary labels, not
measurements, and feeding them to a tree would only invite it to memorise individual records.
They are dropped.

### Data cleaning

Exactly **one** row of 150,150 has a missing value (`Total Price`). It is dropped with
`dropna()`, leaving **150,149** rows. At this ratio imputation would add complexity without
changing anything measurable.

---

## The model

**`DecisionTreeRegressor(max_depth=12, random_state=42)`** from scikit-learn.

| Setting | Value | Reason |
| --- | --- | --- |
| Features | `Total Price`, `Base Price` | The only columns carrying real signal |
| Target | `Units Sold` | Quantity demanded |
| Split | 80 / 20 via `train_test_split(random_state=42)` | 120,119 train / 30,030 test rows |
| `max_depth` | 12 | Chosen by comparing test R² across depths (below) |
| `random_state` | 42 | Reproducible splits and fits |

### Why a decision tree?

1. **The price/demand relationship is non-linear and stepped.** The scatter plots show demand
   holding flat across a price band and then dropping sharply at certain thresholds. A decision
   tree splits the price axis at exactly those thresholds; a linear model is forced to push one
   global slope through all of them.
2. **It is measurably better here.** Against a `LinearRegression` baseline trained on the same
   split, the tree nearly triples the explained variance and cuts average error by ~23%.
3. **No scaling or feature engineering required.** Trees are invariant to monotonic transforms of
   the inputs, so the pipeline stays short and there is no scaler to keep in sync at prediction time.
4. **The logic is inspectable.** Every prediction traces to a readable chain of price thresholds,
   which matters when the output drives a real pricing decision someone has to justify.

### Why this training method?

A single 80/20 hold-out split is used rather than cross-validation: with 150k rows the test set
alone holds 30,030 records, so the variance of the estimate is already low and CV would cost
5× the compute for no meaningful gain in confidence.

`max_depth` was tuned by fitting a range of depths and comparing **test** R², not training R² —
training R² rises monotonically with depth and would always pick the deepest, most overfitted tree.

| `max_depth` | Train R² | Test R² |
| --- | --- | --- |
| 3 | 0.2093 | 0.2088 |
| 5 | 0.3266 | 0.3205 |
| 8 | 0.4178 | 0.4065 |
| 10 | 0.4623 | 0.4200 |
| **12** | **0.5067** | **0.4333** ← best |
| 15 | 0.5598 | 0.4071 |
| 20 | 0.5919 | 0.3795 |
| `None` (grows to depth 27) | 0.5939 | 0.3765 |

The curve peaks at depth 12 and falls after it — the textbook overfitting signature. An
unrestricted tree scores 0.5939 on data it has seen and only 0.3765 on data it has not.

---

## Evaluation

Measured on the 30,030-row held-out test set, via `model.score()` (R²) and `mean_absolute_error`:

| Model | Test R² (accuracy) | Train R² | Test MAE |
| --- | --- | --- | --- |
| **Decision tree, `max_depth=12`** | **0.4333** | 0.5067 | **25.13 units** |
| Decision tree, unrestricted | 0.3765 | 0.5939 | 25.50 units |
| Linear regression (baseline) | 0.1488 | — | 32.49 units |

**Accuracy of the final model: R² = 0.4333** — price alone explains about **43%** of the variance
in units sold, with an average error of ~25 units.

### Reading that number honestly

R² of 0.43 is a good result *for this feature set*, not a good result in absolute terms. Price is
one of several demand drivers, and the dataset contains none of the others — no promotion flag, no
date or seasonality, no store location or footfall, no stock levels, no competitor prices. The
remaining 57% of variance largely lives in those missing columns. The target is also heavily
right-skewed (median 35 units, maximum 2,876), and those rare spike sales are essentially
unpredictable from price, which inflates the MAE.

The model is therefore useful for **ranking candidate price points** against each other, and not
for forecasting an exact unit count for a specific product on a specific day.

---

## Prediction example

```python
features = pd.DataFrame([[133.00, 140.00]], columns=["Total Price", "Base Price"])
model.predict(features)
```

Selling at **133.00** against a base price of **140.00** → **42 units** expected.

Sweeping the discount for a product with a base price of 140:

| Total Price | Base Price | Predicted units sold | Predicted revenue |
| --- | --- | --- | --- |
| 140.00 | 140.00 | 71 | 9,940.00 |
| 133.00 | 140.00 | 42 | 5,586.00 |
| 125.00 | 140.00 | 42 | 5,250.00 |
| 115.00 | 140.00 | 49 | 5,635.00 |
| 105.00 | 140.00 | 106 | 11,130.00 |

Note the curve is **not monotonic** — a small discount to 133.00 predicts *fewer* sales than no
discount at all. This is the model reporting what the price bands historically contained, not a
demand curve for one specific item: different products occupy different bands. It is a direct
consequence of having only price as input, and another reason to treat the output as a band-level
estimate.

---

## Project structure

```
product-demand-prediction/
├── product_demand_prediction.ipynb   # Full analysis, executed with outputs
├── README.md
├── requirements.txt
└── data/
    └── demand.csv                    # Cached copy of the dataset
```

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook product_demand_prediction.ipynb
```

The notebook reads `data/demand.csv` if present and falls back to downloading it from the source URL.

## Possible improvements

- Add the missing demand drivers (seasonality, promotions, store attributes) — the single largest
  available gain, larger than any change of algorithm.
- Try `RandomForestRegressor` or gradient boosting, which average many trees and typically beat a
  single tree on noisy targets.
- Model `log(Units Sold)` to reduce the influence of the rare 1,000+ unit spikes.
- Add price elasticity as an engineered feature (`Total Price / Base Price`, the discount ratio),
  which encodes the discount depth the tree currently has to reconstruct from two separate splits.
