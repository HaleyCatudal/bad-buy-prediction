"""
prep.py
Shared data loading and cleaning for every analysis script.

Every other script imports load_raw() or load_clean() from this file, so the
cleaning rules live in one place. Run this file on its own to print a summary
of each cleaning step.
"""
import pandas as pd

# Path to the Kaggle file (see README for the download link)
FILE = "data/training.csv"

# Predictor lists used across the scripts
NUMERIC = ["VehicleAge", "VehOdo", "MMRAuctionAvg", "CostMinusMMR", "WarrantyCost"]
CATEGORY_ALL = ["WheelType", "Make", "Size", "Auction", "VNST", "BYRNO",
                "PurchMonth", "PurchYear", "Trim", "Color"]          # 15 candidates in total
CATEGORY_FINAL = ["WheelType", "Make", "Size", "Auction", "VNST", "BYRNO"]   # 11 predictors in total

# Time-based split boundaries
VALID_START = "2010-04-01"   # threshold and feature choices are made on Apr-Jun 2010
TEST_START = "2010-07-01"    # test set: Jul-Dec 2010


def load_raw():
    """Read the file with every column as text, so "NULL" entries stay visible."""
    return pd.read_csv(FILE, dtype=str, keep_default_na=False)


def missing_flags(df):
    """One True/False column per missing-value check, on the raw text data."""
    mmr = pd.to_numeric(df["MMRAcquisitionAuctionAveragePrice"], errors="coerce")
    return pd.DataFrame({
        "Primeunit / Aucguart": df["PRIMEUNIT"] == "NULL",
        "Wheel Type": df["WheelType"] == "NULL",
        "Trim": df["Trim"].isin(["", "NULL"]),
        "MMR blank or $0": mmr.isna() | (mmr == 0),
        "Color": df["Color"].isin(["NULL", "NOT AVAIL"]),
        "Small fields": (df["Transmission"].isin(["NULL", ""])
                         | (df["SubModel"] == "NULL")
                         | (df["Size"] == "NULL")),
    })


def load_clean():
    """Apply every cleaning rule and return the analysis-ready data."""
    df = load_raw()
    flags = missing_flags(df)
    cost = pd.to_numeric(df["VehBCost"], errors="coerce")

    # Drop rows: random missing MMR or small fields, and impossible purchase costs
    keep = ~flags["MMR blank or $0"] & ~flags["Small fields"] & (cost >= 1000)
    df = df[keep].copy()

    # Data types
    df["IsBadBuy"] = df["IsBadBuy"].astype(int)
    df["PurchDate"] = pd.to_datetime(df["PurchDate"], format="%m/%d/%Y")
    df["PurchMonth"] = df["PurchDate"].dt.month.astype(str)
    df["PurchYear"] = df["PurchDate"].dt.year.astype(str)

    # Standardize labels and keep systematic gaps as "Missing"
    df["Transmission"] = df["Transmission"].str.upper()
    df["Make"] = df["Make"].replace("TOYOTA SCION", "SCION")
    df["WheelType"] = df["WheelType"].replace("NULL", "Missing")
    df["Trim"] = df["Trim"].replace({"": "Missing", "NULL": "Missing"})
    df["Color"] = df["Color"].replace({"NULL": "Missing", "NOT AVAIL": "Missing"})

    # Numeric predictors, including the derived Cost minus MMR
    df["VehicleAge"] = pd.to_numeric(df["VehicleAge"])
    df["VehYear"] = pd.to_numeric(df["VehYear"])
    df["VehOdo"] = pd.to_numeric(df["VehOdo"])
    df["VehBCost"] = pd.to_numeric(df["VehBCost"])
    df["WarrantyCost"] = pd.to_numeric(df["WarrantyCost"])
    for c in [c for c in df.columns if c.startswith("MMR")]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["MMRAuctionAvg"] = df["MMRAcquisitionAuctionAveragePrice"]
    df["CostMinusMMR"] = df["VehBCost"] - df["MMRAuctionAvg"]

    # Drop columns: 95% empty and rarely recorded before mid-2010
    df = df.drop(columns=["PRIMEUNIT", "AUCGUART"])

    # Drop columns: MMR Current prices are recorded after purchase (leakage)
    df = df.drop(columns=[c for c in df.columns if c.startswith("MMRCurrent")])
    return df


def split(df):
    """Return the training period, its fit and validation parts, and the test period."""
    train = df[df["PurchDate"] < TEST_START]
    test = df[df["PurchDate"] >= TEST_START]
    fit_part = train[train["PurchDate"] < VALID_START]
    valid_part = train[train["PurchDate"] >= VALID_START]
    return train, fit_part, valid_part, test


if __name__ == "__main__":
    raw = load_raw()
    flags = missing_flags(raw)
    cost = pd.to_numeric(raw["VehBCost"], errors="coerce")
    clean = load_clean()

    print("DATA CLEANING SUMMARY")
    print("=" * 46)
    print(f"Rows in training.csv:              {len(raw):,}")
    print(f"Duplicate RefIds:                  {raw['RefId'].duplicated().sum()}")
    print(f"Duplicate rows (ignoring RefId):   {raw.drop(columns='RefId').duplicated().sum()}")
    print(f"Rows dropped, MMR blank or $0:     {flags['MMR blank or $0'].sum():,}")
    print(f"Rows dropped, small fields:        {flags['Small fields'].sum():,}")
    print(f"Rows dropped, purchase cost < $1,000: {(cost < 1000).sum()}")
    print(f"Rows after cleaning:               {len(clean):,}")
    print(f"Good buys after cleaning:          {(clean['IsBadBuy'] == 0).sum():,}")
    print(f"Bad buys after cleaning:           {(clean['IsBadBuy'] == 1).sum():,}")
    print(f"Bad-buy rate after cleaning:       {clean['IsBadBuy'].mean():.1%}")
    print(f"Accuracy of labeling every car good: {1 - clean['IsBadBuy'].mean():.1%}")
    print(f"Average purchase cost:             ${clean['VehBCost'].mean():,.0f}")
