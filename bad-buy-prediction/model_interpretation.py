"""
model_interpretation.py
What the final model learned and how well it generalizes.

1. Overfitting check: AUC on the training, validation, and test periods.
2. Baseline: AUC of a rule that flags only vehicles with a missing wheel type.
3. Calibration: average predicted probability vs. actual bad-buy rate, by risk decile.
4. Coefficients: numeric predictors as odds ratios per standard deviation; Wheel Type
   levels relative to Alloy; the largest State and Buyer effects.
5. Ranking: share of all bad buys caught when inspecting the riskiest X% of vehicles.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from bad_buy_model import model, train, test, valid_part, valid_prob, prob, COLS, NUMERIC

y_test = test["IsBadBuy"].to_numpy()
coef = pd.Series(model.named_steps["logit"].coef_[0],
                 index=model.named_steps["prep"].get_feature_names_out())

print("1. OVERFITTING CHECK (AUC)")
print("=" * 60)
train_auc = roc_auc_score(train["IsBadBuy"], model.predict_proba(train[COLS])[:, 1])
print(f"  Training (Jan 2009 - Jun 2010):     {train_auc:.3f}")
print(f"  Validation (Apr - Jun 2010):        {roc_auc_score(valid_part['IsBadBuy'], valid_prob):.3f}")
print(f"  Test (Jul - Dec 2010):              {roc_auc_score(y_test, prob):.3f}")

print("\n2. BASELINE: FLAG ONLY MISSING WHEEL TYPE")
print("=" * 60)
rule = (test["WheelType"] == "Missing").astype(int)
print(f"  Test AUC of the rule:               {roc_auc_score(y_test, rule):.3f}")
print(f"  Test AUC of the model:              {roc_auc_score(y_test, prob):.3f}")

print("\n3. CALIBRATION ON THE TEST PERIOD")
print("=" * 60)
print(f"  Average predicted probability:      {prob.mean():.1%}")
print(f"  Actual bad-buy rate:                {y_test.mean():.1%}")
deciles = pd.qcut(prob, 10, labels=range(1, 11))
table = pd.DataFrame({"Predicted": prob, "Actual": y_test}).groupby(deciles, observed=True).mean()
print("  Risk decile (10 = riskiest)   Predicted   Actual")
for d, row in table.iterrows():
    print(f"  {d:>4}                          {row['Predicted']:>8.1%}  {row['Actual']:>7.1%}")

print("\n4. COEFFICIENTS")
print("=" * 60)
print("Numeric predictors (per one standard deviation increase, others held fixed):")
for c in NUMERIC:
    b = coef[f"num__{c}"]
    sd = train[c].std()
    print(f"  {c:<15} coefficient {b:+.3f}   odds ratio {np.exp(b):.2f}   (1 SD = {sd:,.1f})")
print("\nWheel Type (odds ratio relative to Alloy):")
alloy = coef["cat__WheelType_Alloy"]
for lvl in ["Covers", "Special", "Missing"]:
    print(f"  {lvl:<10} {np.exp(coef[f'cat__WheelType_{lvl}'] - alloy):.2f}")
for prefix, label in [("cat__VNST_", "State"), ("cat__BYRNO_", "Buyer")]:
    sub = coef[[i for i in coef.index if i.startswith(prefix)]]
    print(f"\n{label}: {len(sub)} encoded levels; coefficients range {sub.min():+.2f} to {sub.max():+.2f}")

print("\n5. BAD BUYS CAUGHT BY INSPECTING THE RISKIEST VEHICLES (test period)")
print("=" * 60)
order = np.argsort(-prob)
for share in [0.10, 0.20, 0.30, (prob >= 0.10).mean()]:
    k = int(round(len(y_test) * share))
    caught = y_test[order[:k]].sum() / y_test.sum()
    print(f"  Riskiest {share:>4.0%} of vehicles   ->   {caught:.0%} of bad buys caught")
