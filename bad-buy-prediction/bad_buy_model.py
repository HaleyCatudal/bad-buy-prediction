"""
bad_buy_model.py
Fits the final logistic regression and evaluates it on the test months.

1. Train on January 2009 to June 2010; test on July to December 2010.
2. Choose the threshold on the held-out April to June 2010 months
   (a preliminary model fit on January 2009 to March 2010), lowering it from
   0.50 until validation recall reaches 70%.
3. Refit on the full training period and score the test months once.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, recall_score, precision_score,
                             accuracy_score, roc_auc_score, fbeta_score)

from prep import load_clean, split, NUMERIC, CATEGORY_FINAL

TARGET_RECALL = 0.70
COLS = NUMERIC + CATEGORY_FINAL


def build_model():
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), CATEGORY_FINAL),
    ])
    # L2 regularization is scikit-learn's default penalty; C = 1.0 sets its strength
    return Pipeline([("prep", prep), ("logit", LogisticRegression(C=1.0, max_iter=2000))])


def choose_threshold(fit_part, valid_part):
    model = build_model().fit(fit_part[COLS], fit_part["IsBadBuy"])
    prob = model.predict_proba(valid_part[COLS])[:, 1]
    for t in np.arange(0.50, 0.01, -0.01):
        if recall_score(valid_part["IsBadBuy"], prob >= t) >= TARGET_RECALL:
            return round(t, 2), model, prob
    return 0.50, model, prob


df = load_clean()
train, fit_part, valid_part, test = split(df)
threshold, valid_model, valid_prob = choose_threshold(fit_part, valid_part)

model = build_model().fit(train[COLS], train["IsBadBuy"])
prob = model.predict_proba(test[COLS])[:, 1]
y = test["IsBadBuy"]


def show_results(label, t):
    pred = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"\n{label}  (threshold = {t:.2f})")
    print("-" * 46)
    print("                      Actual Bad   Actual Good")
    print(f"  Predicted Bad       TP {tp:>6,}    FP {fp:>6,}")
    print(f"  Predicted Good      FN {fn:>6,}    TN {tn:>6,}")
    print()
    print(f"  Accuracy:   {accuracy_score(y, pred):.2f}")
    print(f"  Precision:  {precision_score(y, pred):.2f}")
    print(f"  Recall:     {recall_score(y, pred):.2f}")
    print(f"  F2 score:   {fbeta_score(y, pred, beta=2):.2f}")
    print(f"  Flagged:    {pred.sum():,} vehicles ({pred.mean():.0%})")


if __name__ == "__main__":
    print("BAD-BUY LOGISTIC REGRESSION")
    print("=" * 46)
    print(f"Predictors ({len(COLS)}): {', '.join(COLS)}")
    print(f"Training vehicles: {len(train):,}  (Jan 2009 - Jun 2010)")
    print(f"Test vehicles:     {len(test):,}  (Jul 2010 - Dec 2010)")
    print(f"Test bad buys:     {y.sum():,} ({y.mean():.1%})")
    print(f"Threshold chosen on Apr-Jun 2010: {threshold:.2f} "
          f"(validation recall {recall_score(valid_part['IsBadBuy'], valid_prob >= threshold):.0%})")
    print(f"AUC on test:       {roc_auc_score(y, prob):.3f}")
    show_results("DEFAULT", 0.50)
    show_results("TUNED FOR RECALL", threshold)
    print("\n" + "=" * 46)
