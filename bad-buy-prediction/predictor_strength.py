"""
predictor_strength.py
Measures each variable's relationship with IsBadBuy on its own (cleaned data).

Categorical variables: chi-square test and Cramer's V (0 = none, 1 = perfect).
Numeric variables: AUC, how often the variable alone ranks a randomly chosen
bad buy above a good buy (0.5 = chance, 1.0 = perfect), with a Mann-Whitney test.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import roc_auc_score
from prep import load_clean

df = load_clean()
y = df["IsBadBuy"]

CATEGORICAL = ["WheelType", "Make", "Model", "Size", "Auction", "VNST", "BYRNO",
               "PurchMonth", "PurchYear", "Trim", "Color", "Transmission",
               "Nationality", "TopThreeAmericanName", "IsOnlineSale"]
NUMBERS = {"Vehicle Age": "VehicleAge", "Vehicle Year": "VehYear", "Odometer": "VehOdo",
           "Purchase Cost": "VehBCost", "Warranty Cost": "WarrantyCost",
           "MMR Auction Avg": "MMRAcquisitionAuctionAveragePrice",
           "MMR Auction Clean": "MMRAcquisitionAuctionCleanPrice",
           "MMR Retail Avg": "MMRAcquisitionRetailAveragePrice",
           "MMR Retail Clean": "MMRAcquisitonRetailCleanPrice",
           "Cost minus MMR": "CostMinusMMR"}


def p_text(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


print("CATEGORICAL PREDICTORS (Cramer's V)")
print("=" * 60)
rows = []
for c in CATEGORICAL:
    table = pd.crosstab(df[c], y)
    chi, p, _, _ = chi2_contingency(table)
    v = np.sqrt(chi / (table.values.sum() * (min(table.shape) - 1)))
    rows.append((c, df[c].nunique(), v, p))
for c, levels, v, p in sorted(rows, key=lambda r: -r[2]):
    print(f"  {c:<22}{levels:>6} levels   V = {v:.2f}   p {p_text(p)}")

print("\nNUMERIC PREDICTORS (AUC)")
print("=" * 60)
rows = []
for name, c in NUMBERS.items():
    x = df[c]
    auc = roc_auc_score(y, x)
    direction = "higher" if auc > 0.5 else "lower"
    p = mannwhitneyu(x[y == 1], x[y == 0]).pvalue
    rows.append((name, max(auc, 1 - auc), direction, x[y == 0].mean(), x[y == 1].mean(), p))
for name, auc, d, good, badm, p in sorted(rows, key=lambda r: -r[1]):
    print(f"  {name:<18} AUC = {auc:.2f}  bad buys {d:<6}  good {good:>9,.0f}  bad {badm:>9,.0f}  p {p_text(p)}")

print("\nKEY PATTERNS")
print("=" * 60)
print("Bad-buy rate by Wheel Type:")
print(df.groupby("WheelType")["IsBadBuy"].mean().map("{:.1%}".format).to_string())
print("\nBad-buy rate by Vehicle Age:")
print(df.groupby("VehicleAge")["IsBadBuy"].mean().map("{:.1%}".format).to_string())
print("\nBad-buy rate by Odometer fifth:")
print(df.groupby(pd.qcut(df["VehOdo"], 5), observed=True)["IsBadBuy"].mean().map("{:.1%}".format).to_string())
print("\nBad-buy rate by Warranty Cost fifth:")
print(df.groupby(pd.qcut(df["WarrantyCost"], 5), observed=True)["IsBadBuy"].mean().map("{:.1%}".format).to_string())

print("\nBad-buy rate range across makes and models (groups with at least 100 vehicles):")
for c in ["Make", "Model"]:
    g = df.groupby(c)["IsBadBuy"].agg(["mean", "size"])
    g = g[g["size"] >= 100].sort_values("mean")
    print(f"  {c}: {len(g)} groups, lowest {g.index[0]} {g['mean'].iloc[0]:.1%}, "
          f"highest {g.index[-1]} {g['mean'].iloc[-1]:.1%}")
