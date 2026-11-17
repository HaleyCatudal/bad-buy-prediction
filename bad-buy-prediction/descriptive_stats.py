"""
descriptive_stats.py
Descriptive statistics for the 15 candidate predictors (cleaned data).

Numeric predictors: mean, median, standard deviation, minimum, maximum.
Categorical predictors: number of levels and the most common levels with shares.
Both tables are also saved as CSV files in tables/.
"""
import os
import pandas as pd
from prep import load_clean, NUMERIC, CATEGORY_ALL

os.makedirs("tables", exist_ok=True)
df = load_clean()

LABELS = {"VehicleAge": "Vehicle Age (years)", "VehOdo": "Odometer (miles)",
          "MMRAuctionAvg": "MMR Auction Average ($)", "CostMinusMMR": "Cost minus MMR ($)",
          "WarrantyCost": "Warranty Cost ($)", "WheelType": "Wheel Type", "Make": "Make",
          "Size": "Size", "Auction": "Auction", "VNST": "State", "BYRNO": "Buyer",
          "PurchMonth": "Purchase Month", "PurchYear": "Purchase Year", "Trim": "Trim",
          "Color": "Color"}

numeric = pd.DataFrame({
    "Predictor": [LABELS[c] for c in NUMERIC],
    "Mean": [df[c].mean() for c in NUMERIC],
    "Median": [df[c].median() for c in NUMERIC],
    "Std. dev.": [df[c].std() for c in NUMERIC],
    "Min": [df[c].min() for c in NUMERIC],
    "Max": [df[c].max() for c in NUMERIC],
})

rows = []
for c in CATEGORY_ALL:
    shares = df[c].value_counts(normalize=True)
    top = ", ".join(f"{k} ({v:.0%})" for k, v in shares.head(3).items())
    rows.append({"Predictor": LABELS[c], "Levels": df[c].nunique(), "Most common (share)": top})
categorical = pd.DataFrame(rows)

numeric.to_csv("tables/descriptive_stats_numeric.csv", index=False)
categorical.to_csv("tables/descriptive_stats_categorical.csv", index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)
print(f"DESCRIPTIVE STATISTICS (n = {len(df):,})")
print("=" * 90)
print("Numeric predictors")
print(numeric.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
print("\nCategorical predictors")
print(categorical.to_string(index=False))
print("\nSaved tables/descriptive_stats_numeric.csv and tables/descriptive_stats_categorical.csv")
