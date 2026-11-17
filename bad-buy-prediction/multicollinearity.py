"""
multicollinearity.py
Checks overlap among the numeric predictors with correlations and VIF.

VIF (variance inflation factor) = 1 / (1 - R^2), where R^2 comes from regressing
one predictor on all the others. Above 5 is a concern; above 10 is severe.
"""
import numpy as np
import pandas as pd
from prep import load_clean

df = load_clean()

BEFORE = {"Vehicle Age": "VehicleAge", "Odometer": "VehOdo", "Purchase Cost": "VehBCost",
          "MMR Auction Avg": "MMRAcquisitionAuctionAveragePrice",
          "MMR Auction Clean": "MMRAcquisitionAuctionCleanPrice",
          "MMR Retail Avg": "MMRAcquisitionRetailAveragePrice",
          "MMR Retail Clean": "MMRAcquisitonRetailCleanPrice",
          "Warranty Cost": "WarrantyCost"}
AFTER = {"Vehicle Age": "VehicleAge", "Odometer": "VehOdo", "MMR Auction Avg": "MMRAuctionAvg",
         "Cost minus MMR": "CostMinusMMR", "Warranty Cost": "WarrantyCost"}


def vif(X):
    out = {}
    for c in X.columns:
        target = X[c].to_numpy()
        others = np.column_stack([np.ones(len(X)), X.drop(columns=c).to_numpy()])
        coef, *_ = np.linalg.lstsq(others, target, rcond=None)
        resid = target - others @ coef
        r2 = 1 - (resid ** 2).sum() / ((target - target.mean()) ** 2).sum()
        out[c] = 1 / (1 - r2)
    return pd.Series(out)


pd.set_option("display.width", 200)
before = pd.DataFrame({k: df[v] for k, v in BEFORE.items()})
after = pd.DataFrame({k: df[v] for k, v in AFTER.items()})

print("CORRELATIONS, ALL NUMERIC PREDICTORS (with Vehicle Year)")
print("=" * 60)
with_year = before.assign(**{"Vehicle Year": df["VehYear"]})
print(with_year.corr().round(2).to_string())

print("\nVIF BEFORE (Vehicle Year left out: it restates Vehicle Age)")
print("-" * 40)
print(vif(before).sort_values(ascending=False).round(1).to_string())

print("\nVIF AFTER (final numeric predictors)")
print("-" * 40)
print(vif(after).sort_values(ascending=False).round(1).to_string())
