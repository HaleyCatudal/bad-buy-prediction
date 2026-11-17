"""
missing_values.py
Tests whether each column's missing values are random or systematic.

For each column, a chi-square test compares the bad-buy rate of rows with the
value missing against rows with the value present (raw data, before cleaning).
p < 0.05: the gap is related to IsBadBuy, so the rows are kept as "Missing".
p >= 0.05: the gap appears random, so the rows are dropped.
"""
import pandas as pd
from scipy.stats import chi2_contingency
from prep import load_raw, missing_flags

raw = load_raw()
y = raw["IsBadBuy"].astype(int)
flags = missing_flags(raw)
dates = pd.to_datetime(raw["PurchDate"], format="%m/%d/%Y")

print("MISSING VALUES: CHI-SQUARE TEST AGAINST IsBadBuy")
print("=" * 86)
print(f"{'Column':<22}{'Rows':>8}{'Share':>8}{'Bad-buy rate (missing vs. not)':>34}{'p-value':>10}  Random?")
for name in flags.columns:
    f = flags[name]
    p = chi2_contingency(pd.crosstab(f, y))[1]
    p_text = "< 0.001" if p < 0.001 else f"{p:.2f}"
    rates = f"{y[f].mean():.1%} vs. {y[~f].mean():.1%}"
    print(f"{name:<22}{f.sum():>8,}{f.mean():>8.1%}{rates:>34}{p_text:>10}  {'Yes' if p >= 0.05 else 'No'}")

print("\nPRIMEUNIT RECORDED, BY PURCHASE QUARTER")
print("-" * 40)
recorded = ~flags["Primeunit / Aucguart"]
by_q = recorded.groupby(dates.dt.to_period("Q")).mean()
for q, v in by_q.items():
    print(f"  {q}   {v:6.1%}")

print("\nPRIMEUNIT RECORDED, BY PURCHASE MONTH")
print("-" * 40)
by_m = recorded.groupby(dates.dt.to_period("M")).agg(["sum", "mean"])
for m, row in by_m.iterrows():
    print(f"  {m}   {int(row['sum']):>5,} records   {row['mean']:6.2%}")
before = recorded[dates < "2010-07-01"].mean()
print(f"\nShare recorded before July 2010: {before:.2%}")
same = ((raw["PRIMEUNIT"] == "NULL") == (raw["AUCGUART"] == "NULL")).all()
print(f"AUCGUART missing on exactly the same rows as PRIMEUNIT: {same}")
