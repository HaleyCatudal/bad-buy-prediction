"""
seasonality_test.py
Tests whether the bad-buy rate changes by month and whether the monthly
pattern repeats across 2009 and 2010 (cleaned data).
"""
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr
from prep import load_clean

df = load_clean()
df["Year"] = df["PurchDate"].dt.year
df["Month"] = df["PurchDate"].dt.month
print("Rows after cleaning:", len(df))

# ---- Test 1: does the bad-buy rate differ by month? ----
month_table = pd.crosstab(df["Month"], df["IsBadBuy"])
chi_month, p_month, dof_month, _ = chi2_contingency(month_table)

# ---- Test 2: does the monthly pattern repeat across years? ----
monthly_rates = df.pivot_table(index="Month", columns="Year",
                               values="IsBadBuy", aggfunc="mean")
r, p_corr = pearsonr(monthly_rates[2009], monthly_rates[2010])

# ---- Test 3: is 2010 different from 2009 overall? ----
year_table = pd.crosstab(df["Year"], df["IsBadBuy"])
chi_year, p_year, _, _ = chi2_contingency(year_table)
rate_2009 = df.loc[df["Year"] == 2009, "IsBadBuy"].mean()
rate_2010 = df.loc[df["Year"] == 2010, "IsBadBuy"].mean()

# ---- Monthly rates (the data behind the line chart) ----
print("\nBAD-BUY RATE BY MONTH")
print("=" * 40)
print(monthly_rates.rename(columns=str).to_string(float_format="{:.1%}".format))

# ---- Results ----
tests = [
    ("Rate differs by month",
     f"Chi-square ({dof_month} df) = {chi_month:.1f}", p_month),
    ("2009 pattern vs. 2010 pattern",
     f"Pearson r = {r:.3f}", p_corr),
    ("2009 vs. 2010 overall",
     f"Bad-buy rate {rate_2009:.1%} vs. {rate_2010:.1%}", p_year),
]

print("\nSEASONALITY TESTS")
print("=" * 40)
for name, stat, p in tests:
    verdict = "Yes" if p < 0.05 else "No"
    print(f"\n{name}")
    print(f"  Statistic:     {stat}")
    p_text = "< 0.001" if p < 0.001 else f"{p:.3f}"
    print(f"  p-value:       {p_text}")
    print(f"  Significant:   {verdict} (cutoff 0.05)")
print("\n" + "=" * 40)
