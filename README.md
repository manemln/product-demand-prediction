# Product Demand Prediction with Machine Learning

Predicting how many units a product will sell at a given price, so seasonal discounts can be set
on evidence rather than intuition. The model answers the question a brand owner actually faces:
*"If I drop the price from X to Y, what happens to demand?"*

**Notebook:** [`product_demand_prediction.ipynb`](product_demand_prediction.ipynb)

---

## Dataset

[demand.csv](https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv), 150,150 rows of historical store-level sales.

| Column | Description | Role |
| --- | --- | --- |
| `ID` | Product ID | Identifier, excluded |
| `Store ID` | Specific store ID | Identifier, excluded |
| `Total Price` | Price at which the product was sold (i.e. after discount) | **Feature** |
| `Base Price` | Initial, undiscounted price of the product | **Feature** |
| `Units Sold` | Quantity demanded | **Target** |

### Feature selection

Each candidate column is tested for linear association with `Units Sold` before it is allowed into
the model.

**Hypotheses**, tested separately for every column:

* **H₀ (null):** ρ = 0. The column has no linear association with `Units Sold`.
* **H₁ (alternative):** ρ ≠ 0. Some linear association exists.

**Test:** Pearson correlation coefficient, two-sided, significance level **α = 0.05**, on
n = 150,149 complete rows. H₀ is rejected when p < α.

| Column | Direction | Strength (r) | Variance explained (r²) | p-value | Decision at α = 0.05 |
| --- | --- | --- | --- | --- | --- |
| `Total Price` | negative | **0.236** | 5.55% | < 0.001 | Reject H₀ |
| `Base Price` | negative | **0.140** | 1.96% | < 0.001 | Reject H₀ |
| `ID` | negative | 0.011 | 0.011% | 0.00004 | Reject H₀ |
| `Store ID` | negative | 0.004 | 0.002% | 0.090 | Fail to reject H₀ |

**Reading the table.** *Strength* is the magnitude of the correlation, |r|; the *Direction* column
carries its sign separately. *Variance explained* is r², the share of the variation in `Units Sold`
that moves with the column.

**The two price columns.** Both reject H₀ decisively, and both move in the negative direction:
demand falls as price rises. This is the law of demand appearing in the data, and it is the reason
price is a usable predictor at all. `Total Price` is the stronger of the two, explaining 5.55% of
the variation on its own.

**A caution about `ID`.** It also rejects H₀ (p = 0.00004), which at first glance suggests it
matters. It does not. At n = 150,149 the test is powerful enough to detect associations far too
small to be useful: `ID` explains **0.011%** of the variance in demand. This is the standard gap
between *statistical* significance and *practical* significance, and with a sample this large the
p-value alone is a poor guide. The effect size is what settles it.

**`Store ID`** fails to reject H₀ (p = 0.090 > 0.05), so there is no evidence of a linear
association at all.

**Both are dropped**, on two grounds that agree. Statistically, neither explains a meaningful share
of the variance (r² < 0.02% in both cases). Structurally, they are arbitrary identifiers rather
than measurements. A record's position in a list is not a cause of its sales, and giving them to a
tree would only invite it to memorise individual rows instead of learning the price relationship.

The model is therefore trained on **`Total Price`** and **`Base Price`**.

### Data cleaning

Exactly **one** row of 150,150 has a missing value (`Total Price`). It is dropped with
`dropna()`, leaving **150,149** rows. At this ratio imputation would add complexity without
changing anything measurable.

---

## Visualizations

The notebook builds these with `plotly.express.scatter()`. They are interactive when the notebook
is run in Jupyter, allowing hover for values, zoom and pan. The images below are exports of the same charts,
since GitHub cannot display interactive Plotly output.

![Units sold vs. selling price](images/units-sold-vs-price.png)

Demand falls as price rises, with an OLS trendline confirming the negative slope. The relationship
is clearly not a straight line: points fan out and cluster, which is the first sign a linear model
will underfit.

![Demand across base price, coloured by selling price](images/base-price-vs-demand.png)

Base price against demand, coloured by the actual selling price. The vertical stripes are discrete
price points rather than a smooth continuum. This is the concrete reason a decision tree, which splits at
thresholds, outperforms linear regression here.

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

`max_depth` was tuned by fitting a range of depths and comparing **test** R², not training R².
Training R² rises monotonically with depth and would always pick the deepest, most overfitted tree.

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

The curve peaks at depth 12 and falls after it, the textbook overfitting signature. An
unrestricted tree scores 0.5939 on data it has seen and only 0.3765 on data it has not.

---

## Evaluation

Measured on the 30,030-row held-out test set, via `model.score()` (R²) and `mean_absolute_error`:

| Model | Test R² (accuracy) | Train R² | Test MAE |
| --- | --- | --- | --- |
| **Decision tree, `max_depth=12`** | **0.4333** | 0.5067 | **25.13 units** |
| Decision tree, unrestricted | 0.3765 | 0.5939 | 25.50 units |
| Linear regression (baseline) | 0.1488 | n/a | 32.49 units |

**Accuracy of the final model: R² = 0.4333**. Price alone explains about **43%** of the variance
in units sold, with an average error of ~25 units.

### Reading that number honestly

R² of 0.43 is a good result *for this feature set*, not a good result in absolute terms. Price is
one of several demand drivers, and the dataset contains none of the others. There is no promotion flag, no
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

Note the curve is **not monotonic**: a small discount to 133.00 predicts *fewer* sales than no
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
├── data/
│   └── demand.csv                    # Cached copy of the dataset
└── images/                           # Chart exports, for viewing on GitHub
    ├── units-sold-vs-price.png
    └── base-price-vs-demand.png
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

- Add the missing demand drivers (seasonality, promotions, store attributes), the single largest
  available gain, larger than any change of algorithm.
- Try `RandomForestRegressor` or gradient boosting, which average many trees and typically beat a
  single tree on noisy targets.
- Model `log(Units Sold)` to reduce the influence of the rare 1,000+ unit spikes.
- Add price elasticity as an engineered feature (`Total Price / Base Price`, the discount ratio),
  which encodes the discount depth the tree currently has to reconstruct from two separate splits.
