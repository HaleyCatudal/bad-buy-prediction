"""
data_quality.py
Dataset profile and data-quality checks.

1. Profile: rows, columns, states, date range.
2. Wrong data types: text placeholders in numeric columns.
3. Inconsistent labels: spelling and case variants of the same category.
4. Duplicates: repeated RefIds or repeated rows.
5. Impossible or contradictory values.
Missing values are tested separately in missing_values.py.
"""
import pandas as pd
from prep import load_raw

raw = load_raw()
dates = pd.to_datetime(raw["PurchDate"], format="%m/%d/%Y")
mmr_cols = [c for c in raw.columns if c.startswith("MMR")]
num = {c: pd.to_numeric(raw[c], errors="coerce")
       for c in mmr_cols + ["VehBCost", "VehOdo", "VehYear", "VehicleAge", "WarrantyCost"]}

print("1. DATASET PROFILE")
print("=" * 60)
print(f"Rows (vehicles):   {len(raw):,}")
print(f"Columns:           {raw.shape[1]}")
print(f"States (VNST):     {raw['VNST'].nunique()}")
print(f"Purchase dates:    {dates.min():%B %d, %Y} to {dates.max():%B %d, %Y}")
print(f"Auctions:          {', '.join(sorted(raw['Auction'].unique()))}")

print("\n2. WRONG DATA TYPES")
print("=" * 60)
placeholder = {c: ((raw[c] == "NULL") | (raw[c] == "")).sum() for c in raw.columns}
placeholder = {c: n for c, n in placeholder.items() if n > 0}
print(f"Columns containing the text \"NULL\" or a blank: {len(placeholder)}")
for c, n in sorted(placeholder.items(), key=lambda kv: -kv[1]):
    print(f"  {c:<38}{n:>8,}")
print("\nMMR price columns stored as text (non-numeric entries per column):")
for c in mmr_cols:
    print(f"  {c:<38}{num[c].isna().sum():>8,}")
print("\nIdentifiers stored as numbers (converted to categories):")
print(f"  BYRNO (buyer ID):  {raw['BYRNO'].nunique()} distinct values")
print(f"  VNZIP1 (ZIP code): {raw['VNZIP1'].nunique()} distinct values")

print("\n3. INCONSISTENT LABELS")
print("=" * 60)
print("Transmission:", raw["Transmission"].value_counts().to_dict())
print("Make, Scion variants:", raw.loc[raw["Make"].str.contains("SCION"), "Make"].value_counts().to_dict())
print("Color, unknown labels:", raw.loc[raw["Color"].isin(["NULL", "NOT AVAIL"]), "Color"].value_counts().to_dict())

print("\n4. DUPLICATES")
print("=" * 60)
print(f"Repeated RefIds:                 {raw['RefId'].duplicated().sum()}")
print(f"Repeated rows (ignoring RefId):  {raw.drop(columns='RefId').duplicated().sum()}")

print("\n5. IMPOSSIBLE OR CONTRADICTORY VALUES")
print("=" * 60)
cost, mmr = num["VehBCost"], num["MMRAcquisitionAuctionAveragePrice"]
low = raw.loc[cost < 1000, ["RefId", "VehBCost", "MMRAcquisitionAuctionAveragePrice"]]
print(f"Purchase cost under $1,000:                  {len(low)} rows (removed)")
print(low.to_string(index=False))
age_gap = (dates.dt.year - num["VehYear"]) != num["VehicleAge"]
print(f"Vehicle age differs from purchase year minus model year: {age_gap.sum()} rows (off by one; harmless)")
print(f"Model year after purchase year:              {(num['VehYear'] > dates.dt.year).sum()}")
avg_clean = (num["MMRAcquisitionAuctionCleanPrice"] < mmr) & (num["MMRAcquisitionAuctionCleanPrice"] > 0)
print(f"Clean price below average price (both > $0): {avg_clean.sum()}")
retail = (num["MMRAcquisitionRetailAveragePrice"] < mmr) & (num["MMRAcquisitionRetailAveragePrice"] > 0)
print(f"Retail price below auction price (both > $0): {retail.sum()}")
print(f"Odometer range:                              {num['VehOdo'].min():,.0f} to {num['VehOdo'].max():,.0f} miles")
print(f"Warranty cost range:                         ${num['WarrantyCost'].min():,.0f} to ${num['WarrantyCost'].max():,.0f}")
extreme = (cost > 3 * mmr) & (mmr > 0)
print(f"Purchase cost above 3x MMR (extreme but plausible, kept): {extreme.sum()}")

print("\n6. REDUNDANT COLUMNS")
print("=" * 60)
pairs = [("WheelTypeID", "WheelType"), ("Make", "Nationality"),
         ("Make", "TopThreeAmericanName"), ("VNZIP1", "VNST"), ("Model", "Make")]
for a, b in pairs:
    sub = raw[(raw[a] != "NULL") & (raw[b] != "NULL")]
    multi = (sub.groupby(a)[b].nunique() > 1).sum()
    print(f"  Each {a} maps to one {b}: {multi == 0} "
          f"({sub[a].nunique()} {a} values, {multi} with more than one {b})")
