"""
figures.py
Draws every figure and saves it to figures/.

  key_predictors.png        bad-buy rate by Wheel Type, Vehicle Age, MMR value, and Odometer
  cost_by_threshold.png     total cost of inspections and missed bad buys by threshold
  missing_values.png        rows missing a value, by column
  primeunit_by_month.png    share of records with PRIMEUNIT recorded, by month
  monthly_bad_buy_rate.png  bad-buy rate by month, 2009 vs. 2010
  correlation_heatmap.png   correlations among the numeric predictors
  class_balance.png         count of good and bad buys
  confusion_matrix.png      definitions of the four confusion-matrix outcomes
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from prep import load_raw, load_clean, missing_flags

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "sans-serif"
INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, RED, GRAY = "#2a78d6", "#eb6834", "#e34948", "#b9b8b3"

raw = load_raw()
df = load_clean()
overall = df["IsBadBuy"].mean()


def style(ax, ygrid=True):
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, colors=MUTED, labelsize=10)


def pct_axis(ax):
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))


def title(ax, text, sub=None, size=15):
    ax.set_title(text, loc="left", fontsize=size, weight="bold", color=INK, pad=26 if sub else 14)
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=10.5, color=MUTED)


def save(fig, name, face=BG):
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight", facecolor=face, pad_inches=0.3)
    plt.close(fig)


def rate_bars(ax, labels, rates, highlight=None):
    colors = [RED if (highlight is not None and i == highlight) else BLUE for i in range(len(rates))]
    ax.bar(range(len(rates)), rates, color=colors, width=0.7)
    for i, r in enumerate(rates):
        ax.text(i, r + 0.008, f"{r:.1%}", ha="center", va="bottom", fontsize=9, color=INK,
                bbox=dict(boxstyle="square,pad=0.1", fc=BG, ec="none"))
    ax.axhline(overall, color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(labels, fontsize=9.5, color=MUTED)
    pct_axis(ax)
    style(ax)


# ---------- Four key predictors ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), dpi=200)
fig.patch.set_facecolor(BG)

order = ["Alloy", "Covers", "Special", "Missing"]
g = df.groupby("WheelType")["IsBadBuy"].mean().reindex(order)
rate_bars(axes[0, 0], order, g.values, highlight=3)
axes[0, 0].set_ylim(0, 0.8)
axes[0, 0].set_title("Wheel Type", loc="left", fontsize=12.5, weight="bold", color=INK)

age = df.groupby("VehicleAge")["IsBadBuy"].agg(["mean", "size"])
age = age[age["size"] >= 100]
rate_bars(axes[0, 1], [str(a) for a in age.index], age["mean"].values)
axes[0, 1].set_ylim(0, 0.38)
axes[0, 1].set_title("Vehicle Age (years)", loc="left", fontsize=12.5, weight="bold", color=INK)

def fifths(col):
    q = pd.qcut(df[col], 5)
    r = df.groupby(q, observed=True)["IsBadBuy"].mean()
    labs = [f"{iv.left:,.0f}-\n{iv.right:,.0f}" for iv in r.index]
    return labs, r.values

labs, r = fifths("MMRAuctionAvg")
labs = ["$" + l.replace("-\n", "-\n$") for l in labs]
labs[0] = f"${df['MMRAuctionAvg'].min():,.0f}-\n" + labs[0].split("-\n")[1]
rate_bars(axes[1, 0], labs, r)
axes[1, 0].set_ylim(0, 0.24)
axes[1, 0].set_title("MMR Auction Average, by fifth", loc="left", fontsize=12.5, weight="bold", color=INK)

labs, r = fifths("VehOdo")
rate_bars(axes[1, 1], labs, r)
axes[1, 1].set_ylim(0, 0.24)
axes[1, 1].set_title("Odometer (miles), by fifth", loc="left", fontsize=12.5, weight="bold", color=INK)

fig.suptitle("Bad-Buy Rate by the Four Strongest Predictors", x=0.065, ha="left", y=1.0,
             fontsize=16, weight="bold", color=INK)
fig.text(0.065, 0.955, f"Dashed line: overall bad-buy rate ({overall:.1%}). n = {len(df):,} vehicles.",
         fontsize=10.5, color=MUTED)
fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.5, w_pad=3)
save(fig, "key_predictors.png")

# ---------- Total cost by threshold ----------
from bad_buy_model import prob, y, threshold   # fits the final model
yt = y.to_numpy()
ts = np.round(np.arange(0.02, 0.51, 0.01), 2)
cost = [(200 * (prob >= t).sum() + 2000 * ((prob < t) & (yt == 1)).sum()) / 1e6 for t in ts]
no_insp, all_insp = 2000 * yt.sum() / 1e6, 200 * len(yt) / 1e6
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
ax.plot(ts, cost, color=BLUE, lw=2.2)
v_best = cost[list(ts).index(round(threshold, 2))]
v_def = cost[list(ts).index(0.50)]
for t, v, text, dx, dy, ha in [(threshold, v_best, f"Threshold {threshold:.2f}\n${v_best:.2f}M (lowest)", 0.012, -0.32, "left"),
                               (0.50, v_def, f"Default 0.50\n${v_def:.2f}M", -0.012, -0.38, "right")]:
    ax.scatter([t], [v], s=70, color=BLUE, edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(text, (t, v), xytext=(t + dx, v + dy), fontsize=11, color=INK, ha=ha)
ax.axhline(no_insp, color=MUTED, lw=1, ls="--")
ax.text(0.505, no_insp, f"No inspections: ${no_insp:.2f}M", va="bottom", ha="right", fontsize=10, color=MUTED)
ax.axhline(all_insp, color=MUTED, lw=1, ls=":")
ax.text(0.505, all_insp + 0.03, f"Inspect every car: ${all_insp:.2f}M", va="bottom", ha="right", fontsize=10, color=MUTED)
ax.set_xlim(0.02, 0.51)
ax.set_ylim(2.5, 5.1)
ax.set_xlabel("Flagging threshold", color=MUTED, fontsize=11)
ax.set_ylabel("Total cost, test period ($ millions)", color=MUTED, fontsize=11)
style(ax)
title(ax, "Total Cost of Inspections and Missed Bad Buys by Threshold",
      f"Assumes \\$200 per inspection and \\$2,000 lost per bad buy bought; {len(yt):,} test-period vehicles")
save(fig, "cost_by_threshold.png")

# ---------- missing values by column ----------
flags = missing_flags(raw)
counts = flags.sum().sort_values()
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
ax.barh(range(len(counts)), counts.values, color=BLUE, height=0.65)
for i, v in enumerate(counts.values):
    ax.text(v + 600, i, f"{v:,}", va="center", fontsize=10, color=INK)
ax.set_yticks(range(len(counts)))
ax.set_yticklabels(counts.index, fontsize=10.5, color=INK)
ax.set_xlim(0, counts.max() * 1.12)
ax.xaxis.set_visible(False)
style(ax, ygrid=False)
ax.spines["bottom"].set_visible(False)
title(ax, "Rows Missing a Value, by Column", f"Raw data, n = {len(raw):,} vehicles")
save(fig, "missing_values.png")

# ---------- PRIMEUNIT recorded by month ----------
month = pd.to_datetime(raw["PurchDate"], format="%m/%d/%Y").dt.to_period("M")
share = (raw["PRIMEUNIT"] != "NULL").groupby(month).mean() * 100
x = share.index.to_timestamp()
fig, ax = plt.subplots(figsize=(10, 5.4), dpi=200)
ax.plot(x, share.values, color=BLUE, lw=2.2)
ax.scatter([x[-1]], [share.values[-1]], s=60, color=BLUE, edgecolor="white", linewidth=2, zorder=3)
ax.annotate(f"{share.values[-1]:.0f}% in Dec 2010", (x[-1], share.values[-1]), xytext=(-12, 0),
            textcoords="offset points", ha="right", va="center", fontsize=11, color=INK)
ax.annotate("Under 0.1% recorded, Jan 2009 - Jun 2010", (x[8], 0), xytext=(0, 10),
            textcoords="offset points", ha="center", fontsize=11, color=MUTED)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylim(-1, share.max() * 1.12)
style(ax)
title(ax, "Percent of Records with a Recorded Value for Primeunit and Aucguart")
save(fig, "primeunit_by_month.png")

# ---------- bad-buy rate by month, 2009 vs. 2010 ----------
monthly = df.assign(Year=df["PurchDate"].dt.year, Month=df["PurchDate"].dt.month) \
            .pivot_table(index="Month", columns="Year", values="IsBadBuy", aggfunc="mean")
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fig, ax = plt.subplots(figsize=(10, 5.4), dpi=200)
for yr, color in [(2009, BLUE), (2010, ORANGE)]:
    ax.plot(range(12), monthly[yr].values, color=color, lw=2.2, marker="o", ms=5)
    ax.text(11.25, monthly[yr].values[-1], str(yr), color=color, fontsize=11, weight="bold", va="center")
ax.axhline(overall, color=MUTED, lw=1, ls="--")
ax.text(-0.3, overall + 0.002, f"Overall {overall:.1%}", fontsize=10, color=MUTED, va="bottom")
ax.set_xticks(range(12))
ax.set_xticklabels(months)
ax.set_xlim(-0.4, 11.9)
ax.set_ylim(0, 0.18)
pct_axis(ax)
style(ax)
title(ax, "Bad-Buy Rate by Purchase Month, 2009 vs. 2010",
      "Rate differs by month (chi-square p < 0.001); 2009 and 2010 patterns correlate at r = 0.53 (p = 0.075)")
save(fig, "monthly_bad_buy_rate.png")

# ---------- correlation heatmap ----------
cols = {"Vehicle Age": "VehicleAge", "Vehicle Year": "VehYear", "Odometer": "VehOdo",
        "Purchase Cost": "VehBCost", "MMR Auction Avg": "MMRAcquisitionAuctionAveragePrice",
        "MMR Auction Clean": "MMRAcquisitionAuctionCleanPrice",
        "MMR Retail Avg": "MMRAcquisitionRetailAveragePrice",
        "MMR Retail Clean": "MMRAcquisitonRetailCleanPrice", "Warranty Cost": "WarrantyCost"}
corr = pd.DataFrame({k: df[v] for k, v in cols.items()}).corr().iloc[1:, :-1]
n = len(corr)
cmap = LinearSegmentedColormap.from_list("d", [BLUE, "#f0efec", RED])
fig, ax = plt.subplots(figsize=(10, 8.4), dpi=200)
im = ax.imshow(np.ma.array(corr.values, mask=np.triu(np.ones((n, n), bool), k=1)), cmap=cmap, vmin=-1, vmax=1)
for i in range(n):
    for j in range(i + 1):
        v = corr.values[i, j]
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=BG, lw=2))
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                color="white" if abs(v) >= 0.75 else INK, weight="bold" if abs(v) >= 0.8 else "normal")
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(corr.columns, rotation=40, ha="right", fontsize=11, color=MUTED)
ax.set_yticklabels(corr.index, fontsize=11, color=MUTED)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_title("Correlation Between Numeric Predictors", fontsize=18, weight="bold", color=INK, loc="left", pad=14)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.outline.set_visible(False)
cb.set_label("Correlation (r)", color=MUTED, fontsize=11)
save(fig, "correlation_heatmap.png")

# ---------- good vs. bad buys ----------
good, bad = (df["IsBadBuy"] == 0).sum(), (df["IsBadBuy"] == 1).sum()
fig, ax = plt.subplots(figsize=(10, 3.2), dpi=200)
ax.barh([1, 0], [good, bad], color=[GRAY, RED], height=0.6)
for yv, v in [(1, good), (0, bad)]:
    ax.text(v + 800, yv, f"{v:,} ({v / len(df):.1%})", va="center", fontsize=11, color=INK)
ax.set_yticks([1, 0])
ax.set_yticklabels(["Good buys", "Bad buys"], fontsize=11, color=INK)
ax.set_xlim(0, good * 1.25)
ax.xaxis.set_visible(False)
style(ax, ygrid=False)
ax.spines["bottom"].set_visible(False)
title(ax, "Good vs. Bad Buys", f"Cleaned data, n = {len(df):,} vehicles")
save(fig, "class_balance.png")

# ---------- confusion matrix definitions ----------
fig, ax = plt.subplots(figsize=(12, 7.6), dpi=200)
ax.set_xlim(0, 12)
ax.set_ylim(0, 7.6)
ax.axis("off")
x0, cw, y_hdr, hh, ch = 2.6, 4.3, 5.3, 1.15, 2.05
ax.text(6, 7.05, "Confusion Matrix: Predicting Bad Buys", ha="center", va="center", fontsize=24, weight="bold", color=INK)
for i, t in enumerate(["Actually\nBad Buy (1)", "Actually\nGood Buy (0)"]):
    ax.add_patch(Rectangle((x0 + i * cw, y_hdr), cw, hh, fill=False, lw=2.5, ec=INK))
    ax.text(x0 + i * cw + cw / 2, y_hdr + hh / 2, t, ha="center", va="center", fontsize=16, color=INK)
cells = [[("True Positive", "Model flags the car as a bad buy,\nand it really is one.\nDealer avoids a lemon.", "#2b6cb0", "white"),
          ("False Positive", "Model flags the car as a bad buy,\nbut it was fine.\nDealer passes on a good car.", "#2b6cb0", "white")],
         [("False Negative", "Model says the car is fine,\nbut it is a lemon.\nDealer buys a lemon.", "#c0392b", "#fdecea"),
          ("True Negative", "Model says the car is fine,\nand it is.\nDealer buys a good car.", "#2b6cb0", "white")]]
for r, row_label in enumerate(["Predicted\nBad Buy (1)", "Predicted\nGood Buy (0)"]):
    yy = y_hdr - (r + 1) * ch
    ax.add_patch(Rectangle((x0 - 2.4, yy), 2.4, ch, fill=False, lw=2.5, ec=INK))
    ax.text(x0 - 1.2, yy + ch / 2, row_label, ha="center", va="center", fontsize=16, color=INK)
    for c in range(2):
        t, desc, color, fill = cells[r][c]
        ax.add_patch(Rectangle((x0 + c * cw, yy), cw, ch, fc=fill, lw=2.5, ec=INK))
        ax.text(x0 + c * cw + cw / 2, yy + ch * 0.72, t, ha="center", va="center", fontsize=18, weight="bold", color=color)
        ax.text(x0 + c * cw + cw / 2, yy + ch * 0.34, desc, ha="center", va="center", fontsize=12.5, color=INK, linespacing=1.4)
ax.text(6, 0.55, "Recall = True Positives / (True Positives + False Negatives)\n"
        "The share of real bad buys the model catches; we minimize false negatives.",
        ha="center", va="center", fontsize=14, color=INK, linespacing=1.6)
save(fig, "confusion_matrix.png", face="white")

print(f"Saved 8 figures to {OUT}/")
