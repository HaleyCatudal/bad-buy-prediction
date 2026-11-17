"""
feature_selection.py
Decides which of the 15 candidate predictors stay in the model.

Rule: keep the smallest set whose validation AUC is within 0.001 of all 15.
Every comparison is made on the validation months (April to June 2010), using a
model fit on January 2009 to March 2010. The test months (July to December 2010)
are never used here, so they stay unseen until the final evaluation.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from prep import load_clean, split, NUMERIC, CATEGORY_ALL, CATEGORY_FINAL


def validation_auc(numeric, category, fit_part, valid_part):
    steps = []
    if numeric:
        steps.append(("num", StandardScaler(), numeric))
    if category:
        steps.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                           min_frequency=50), category))
    model = Pipeline([("prep", ColumnTransformer(steps)),
                      ("logit", LogisticRegression(max_iter=2000))])
    cols = numeric + category
    model.fit(fit_part[cols], fit_part["IsBadBuy"])
    prob = model.predict_proba(valid_part[cols])[:, 1]
    return roc_auc_score(valid_part["IsBadBuy"], prob)


df = load_clean()
_, fit_part, valid_part, _ = split(df)

full = validation_auc(NUMERIC, CATEGORY_ALL, fit_part, valid_part)

print("FEATURE SELECTION (validation months: Apr-Jun 2010)")
print("=" * 56)
print(f"Fit rows: {len(fit_part):,}   Validation rows: {len(valid_part):,}")
print(f"\nAll 15 candidates                       AUC = {full:.4f}")

print("\nRemoving one predictor at a time:")
for c in NUMERIC:
    auc = validation_auc([x for x in NUMERIC if x != c], CATEGORY_ALL, fit_part, valid_part)
    print(f"  without {c:<16}  AUC = {auc:.3f}   change = {auc - full:+.3f}")
for c in CATEGORY_ALL:
    auc = validation_auc(NUMERIC, [x for x in CATEGORY_ALL if x != c], fit_part, valid_part)
    print(f"  without {c:<16}  AUC = {auc:.3f}   change = {auc - full:+.3f}")

BASE = ["WheelType", "Make", "Size", "Auction"]
sets = [
    ("All 15 candidates", CATEGORY_ALL),
    ("Base 9 + State + Buyer + Trim + Color", BASE + ["VNST", "BYRNO", "Trim", "Color"]),
    ("Base 9 + State + Buyer + Month + Year", BASE + ["VNST", "BYRNO", "PurchMonth", "PurchYear"]),
    ("Base 9 + State + Buyer (final 11)", CATEGORY_FINAL),
    ("Base 9 + Buyer", BASE + ["BYRNO"]),
    ("Base 9 + State", BASE + ["VNST"]),
    ("Base 9", BASE),
]
print("\nCandidate sets (base 9 = 5 numeric + Wheel Type, Make, Size, Auction):")
for name, cats in sets:
    auc = validation_auc(NUMERIC, cats, fit_part, valid_part)
    print(f"  {name:<40} {len(NUMERIC) + len(cats):>2} predictors   AUC = {auc:.4f}   gap to all 15 = {full - auc:.4f}")
core4 = validation_auc(["VehicleAge", "VehOdo", "MMRAuctionAvg"], ["WheelType"], fit_part, valid_part)
print(f"  {'Wheel Type, Age, Value, Odometer only':<40}  4 predictors   AUC = {core4:.4f}   gap to all 15 = {full - core4:.4f}")
print("\nRule: keep the smallest set within 0.001 of all 15 -> final 11 predictors.")
