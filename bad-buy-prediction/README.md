# Predicting Bad Buys at Used-Car Auctions

Logistic regression that flags auction vehicles likely to be bad buys, so a dealer can send them for a pre-purchase inspection before bidding.

Business Intelligence and Analytics final project, University of Georgia, Fall 2026.
Team: Owen Williams, Haley Catudal, Abem Kebede, Daniel Song.

## Data

The data comes from the Kaggle competition ["Don't Get Kicked!"](https://www.kaggle.com/c/DontGetKicked), hosted by Carvana. The file is not included in this repository; download it from Kaggle after accepting the competition rules.

1. Download `training.csv` from the competition's Data tab.
2. Place it in the `data/` folder of this repository.

One row is one vehicle purchased at auction: 72,983 vehicles, 34 columns, January 2009 to December 2010. The target is `IsBadBuy` (1 = bad buy, 0 = good buy).

Column definitions come from the competition's data dictionary ([formatted copy](https://users.phhp.ufl.edu/rlp176/Courses/PHC6089/data/Carvana_Data_Dictionary_formatted.txt)). It defines the MMR Acquisition prices as prices at the time of purchase and the MMR Current prices as prices "as of current day," which is why the MMR Current prices are excluded.

## Setup

Python 3.10 or later.

```
pip install -r requirements.txt
```

## Scripts

Run each script from the repository folder (for example, `python bad_buy_model.py`, or open it in IDLE and press F5). `prep.py` holds the cleaning rules and file path; every other script imports it.

| Script | Output |
| --- | --- |
| `prep.py` | Cleaning summary: rows dropped by rule, rows remaining, good and bad buy counts, bad-buy rate, accuracy of labeling every vehicle good, average purchase cost. |
| `data_quality.py` | Dataset profile (rows, columns, states, date range, auctions) and the quality checks: text placeholders, inconsistent labels, duplicates, impossible values, extreme values. |
| `missing_values.py` | Rows and share missing per column, chi-square test of each against `IsBadBuy`, and PRIMEUNIT recorded by quarter and by month. |
| `descriptive_stats.py` | Mean, median, standard deviation, minimum, and maximum for numeric predictors; levels and most common values for categorical predictors. Saved to `tables/`. |
| `predictor_strength.py` | Cramér's V for categorical variables, AUC for numeric variables, and the bad-buy rates by Wheel Type, Vehicle Age, Odometer, Warranty Cost, make, and model. |
| `seasonality_test.py` | Bad-buy rate by month for 2009 and 2010, and the three seasonality tests. |
| `multicollinearity.py` | Correlation matrix and VIF before and after removing redundant variables. |
| `feature_selection.py` | Validation AUC for each candidate predictor set and the selection rule. |
| `bad_buy_model.py` | Final model: chosen threshold, test AUC, and confusion matrix, accuracy, precision, recall, F2, and vehicles flagged at 0.50 and at the tuned threshold. |
| `model_interpretation.py` | Training, validation, and test AUC; a missing-wheel-type baseline; calibration by risk decile; coefficients as odds ratios; share of bad buys caught when inspecting the riskiest vehicles. |
| `cost_analysis.py` | Total cost of each strategy, savings, the break-even probability, and the threshold sensitivity table. |
| `figures.py` | Every figure, saved to `figures/`. |

## Figures

| File | Shows |
| --- | --- |
| `figures/key_predictors.png` | Bad-buy rate by Wheel Type, Vehicle Age, MMR Auction Average, and Odometer |
| `figures/cost_by_threshold.png` | Total cost of inspections and missed bad buys by threshold |
| `figures/missing_values.png` | Rows missing a value, by column |
| `figures/primeunit_by_month.png` | Share of records with PRIMEUNIT recorded, by month |
| `figures/monthly_bad_buy_rate.png` | Bad-buy rate by month, 2009 vs. 2010 |
| `figures/correlation_heatmap.png` | Correlations among the numeric predictors |
| `figures/class_balance.png` | Count of good and bad buys |
| `figures/confusion_matrix.png` | Definitions of the four confusion-matrix outcomes |

## Where each result comes from

| Topic | Results | Script |
| --- | --- | --- |
| Dataset | 72,983 vehicles, 34 columns, 37 states, January 2009 to December 2010 | `data_quality.py` |
| Data quality | Text placeholders in 19 columns; "Manual" and "TOYOTA SCION" variants; no duplicates; purchase costs of $1 and $225; 32 vehicles bought above 3x MMR | `data_quality.py` |
| Cleaning | 72,123 vehicles remain; 63,265 good and 8,858 bad buys (12.3%); 87.7% accuracy from labeling every vehicle good; $6,730 average purchase cost | `prep.py` |
| Missing values | Chi-square results per column; PRIMEUNIT and AUCGUART 95% empty, under 0.1% recorded before July 2010 | `missing_values.py` |
| Descriptive statistics | Mean, median, standard deviation, and frequencies for the 15 candidate predictors | `descriptive_stats.py` |
| Predictor strength | Cramér's V and AUC for every variable; Wheel Type 70.5% vs. 8.0% to 13.2%; Vehicle Age 4.1% to 31.4%; MMR $5,481 vs. $6,300; Odometer 8.4% to 16.4%; makes 6.8% to 20.3%; Transmission and IsOnlineSale p = 0.40 | `predictor_strength.py` |
| Seasonality | Monthly rates; month chi-square p < 0.001; 2009 vs. 2010 r = 0.53, p = 0.075; 11.5% vs. 13.0%, p < 0.001 | `seasonality_test.py` |
| Multicollinearity | MMR correlations 0.89 to 0.99; purchase cost 0.82; Vehicle Year -0.96; VIF up to 158 before, below 2.5 after | `multicollinearity.py` |
| Feature selection | Validation AUC 0.755 for 15 and 11 predictors, 0.748 for 9 | `feature_selection.py` |
| Model | 52,792 training and 19,331 test vehicles; threshold 0.10 (74% validation recall); test AUC 0.75; confusion matrices and metrics at 0.50 and 0.10 | `bad_buy_model.py` |
| Model interpretation | AUC 0.756 training, 0.755 validation, 0.751 test; missing-wheel-type rule AUC 0.618; average predicted 11.3% vs. actual 12.5%; Vehicle Age odds ratio 1.65 per SD; Cost minus MMR 0.80 per SD; missing wheel type 22.9 times the odds of Alloy; riskiest 10% and 20% of vehicles hold 37% and 51% of bad buys | `model_interpretation.py` |
| Cost | Strategy costs; $1,881,200 and $909,400 savings; $2,000 as 30% of $6,730; break-even 0.10; sensitivity table | `cost_analysis.py` |

The $100 to $250 inspection cost range comes from [Auto Credit Express](https://www.autocreditexpress.com/blog/pre-purchase-inspections-are-key-when-buying-a-used-car/); the $200 inspection cost and $2,000 loss per bad buy are assumptions.

## Cleaning rules

- Rows with a blank or $0 MMR auction average price (846) or blanks in Transmission, SubModel, or Size (14) are dropped; their missingness is unrelated to `IsBadBuy`.
- Rows with a purchase cost under $1,000 (2) are dropped as impossible values.
- Blank Wheel Type, Trim, and Color values are kept as a "Missing" category; their missingness is related to `IsBadBuy`.
- PRIMEUNIT and AUCGUART are dropped; they are 95% empty, and under 0.1% of records had them before July 2010.
- The four MMR Current prices are dropped; they are recorded after purchase.
- 72,123 vehicles remain.

## Model

- Logistic regression (scikit-learn, L2 regularization, C = 1).
- 11 predictors: Vehicle Age, Odometer, MMR Auction Average, Cost minus MMR, Warranty Cost, Wheel Type, Make, Size, Auction, State, Buyer.
- Numeric predictors are standardized to z-scores; categorical predictors are one-hot encoded, with levels under 50 training observations pooled.
- Training: January 2009 to June 2010 (52,792 vehicles). Test: July to December 2010 (19,331 vehicles).
- Feature selection and the threshold are chosen on April to June 2010, using a model fit on January 2009 to March 2010. The test months are scored once.

## Results (test months)

| Threshold | Recall | Precision | Vehicles flagged | AUC |
| --- | --- | --- | --- | --- |
| 0.50 (default) | 22% | 87% | 3% | 0.75 |
| 0.10 (tuned) | 68% | 24% | 36% | 0.75 |

Assuming $200 per inspection and $2,000 per bad buy, inspecting every vehicle scored above 0.10 lowers total cost from $4,838,000 to $2,956,800 (39%).

## Tableau

The descriptive charts were also built in Tableau during exploration. To rebuild them:

1. Connect to `training.csv` as a text file.
2. Convert Byrno, Vnzip1, Veh Year, and Is Online Sale to dimensions; change Byrno and Vnzip1 to strings.
3. Assign geographic roles: State/Province to Vnst, ZIP Code/Postcode to Vnzip1.
4. Set the default aggregation of Is Bad Buy to Average and its number format to Percentage.
5. Change Veh B Cost to Number (decimal).
6. Group "Manual" with "MANUAL" in Transmission and "TOYOTA SCION" with "SCION" in Make; alias the null values of Wheel Type, Trim, and Color as "Missing."
7. Create one calculated field per missing-value check, for example `IF ISNULL([Wheel Type]) OR [Wheel Type]="NULL" THEN 1 ELSE 0 END`.
8. Create `Keep Row` as `[Missing: MMR] = 0 AND [Missing: Small Fields] = 0 AND [Veh B Cost] >= 1000` and apply it as a filter set to True on every sheet except the missing-value charts.
9. Missing values by column: drag Measure Values to Columns and Measure Names to Rows, keep only the step 7 fields, and set each to SUM.
10. Good and bad buys: duplicate Is Bad Buy, convert the copy to a dimension, alias 0 as "Good Buy" and 1 as "Bad Buy," place it on Columns, and place training.csv (Count) on Rows.
11. Bad-buy rate by month: place MONTH(Purch Date) on Columns, AVG(Is Bad Buy) on Rows, and YEAR(Purch Date) on Color.
