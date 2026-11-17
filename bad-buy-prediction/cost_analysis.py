"""
cost_analysis.py
Converts model results into dollars and finds the cost-minimizing threshold.

Assumptions: each inspection costs $200; each bad buy that is bought costs $2,000;
an inspection catches every bad buy it is applied to.
Total cost = $200 x vehicles inspected + loss per bad buy x bad buys missed.

The best threshold for each assumed loss is chosen on the validation months
(April to June 2010) and then applied to the test months.
"""
import numpy as np
from bad_buy_model import valid_part, valid_prob, prob, y, threshold, df

INSPECTION = 200
LOSS = 2000
THRESHOLDS = np.round(np.arange(0.02, 0.51, 0.01), 2)
yv = valid_part["IsBadBuy"].to_numpy()
yt = y.to_numpy()


def total_cost(actual, p, t, loss):
    flagged = p >= t
    missed = (~flagged & (actual == 1)).sum()
    return INSPECTION * flagged.sum() + loss * missed, flagged, missed


def money(x):
    return f"${x:,.0f}"


n = len(yt)
bad = yt.sum()
print("COST ANALYSIS (test months: Jul-Dec 2010)")
print("=" * 70)
print(f"Assumptions: ${INSPECTION:,} per inspection, ${LOSS:,} per bad buy bought")
avg_cost = df["VehBCost"].mean()
print(f"${LOSS:,} is {LOSS / avg_cost:.0%} of the average purchase cost (${avg_cost:,.0f}).")
print(f"Break-even probability: ${INSPECTION:,} / ${LOSS:,} = {INSPECTION / LOSS:.2f}\n")
print(f"{'Strategy':<26}{'Inspections':>14}{'Bad-buy losses':>16}{'Total':>14}")
rows = [("No inspections", 0, LOSS * bad),
        ("Inspect every vehicle", INSPECTION * n, 0)]
for t in [0.50, threshold]:
    c, f, m = total_cost(yt, prob, t, LOSS)
    rows.append((f"Model at {t:.2f}", INSPECTION * f.sum(), LOSS * m))
for name, i, l in rows:
    print(f"{name:<26}{money(i):>14}{money(l):>16}{money(i + l):>14}")
best_total = rows[-1][1] + rows[-1][2]
print(f"\nSavings vs. no inspections:   {money(rows[0][2] - best_total)} "
      f"({(rows[0][2] - best_total) / rows[0][2]:.0%})")
print(f"Savings vs. inspecting all:   {money(rows[1][1] - best_total)}")

print("\nSENSITIVITY: best threshold for each assumed loss per bad buy")
print("-" * 70)
print(f"{'Loss per bad buy':<18}{'Threshold':>10}{'Recall':>9}{'Inspected':>11}{'Total cost':>14}")
for loss in [1000, 2000, 3000, 5000]:
    t = min(THRESHOLDS, key=lambda th: total_cost(yv, valid_prob, th, loss)[0])
    c, f, m = total_cost(yt, prob, t, loss)
    print(f"{money(loss):<18}{t:>10.2f}{(bad - m) / bad:>9.0%}{f.mean():>11.0%}{money(c):>14}")
