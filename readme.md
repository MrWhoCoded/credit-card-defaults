# Credit Card Default Prediction

Predicting credit card defaults using the UCI Credit Card dataset. This project focuses on thoughtful feature engineering and threshold-based decision-making, not just model accuracy.

## Why This Problem is Tricky

The default rate in this dataset is around 22%. Sounds small, but in credit risk, missing a defaulter is costly. A model that just predicts "no default" for everyone still gets ~78% accuracy — but catches zero actual defaults. That's useless.

So accuracy doesn't work here. What matters more is **recall on the minority class** — catching the people who will actually default, even at the cost of some false alarms.

## What I Found in the Data

The dataset has 6 months of payment history per customer: repayment status (PAY_0 through PAY_6), bill amounts (BILL_AMT1-6), and payment amounts (PAY_AMT1-6). Plotting a correlation heatmap immediately showed the issue — billing amounts across months are highly correlated with each other. Same with payments.

This temporal redundancy doesn't add signal, it just adds noise and makes the model heavier. So rather than running PCA blindly, I aggregated these into summary features that actually capture behavior:

- `max_delay` / `avg_delay` — worst and average repayment status across months
- `max_bill` / `avg_bill` — spending pattern
- `bill_trend` — difference between most recent and oldest bill (are they paying down or accumulating?)
- `max_pay_amt` / `avg_pay_amt` — how much they actually pay

Then I dropped the individual month-level columns. Fewer features, cleaner signal.

## What I Dropped (and Why)

- Individual PAY_2 through PAY_6 — replaced by aggregated delay features
- BILL_AMT2 through BILL_AMT6 — high multicollinearity, replaced by aggregates
- PAY_AMT2 through PAY_AMT6 — same logic
- ID — not a feature
- Demographic columns (SEX, EDUCATION, MARRIAGE, AGE) — kept out of the final model; they didn't help much and raised fairness questions

The goal was to reduce redundancy without losing predictive power.

## Handling the Imbalance

I didn't resample. SMOTE or random oversampling can create problems — leakage in cross-validation, synthetic samples that don't reflect reality. Instead:

1. **Class weights** — used `class_weight="balanced"` in Random Forest, which internally adjusts for class frequency
2. **Threshold tuning** — instead of using 0.5 as the decision boundary for predicted probabilities, I tested multiple thresholds

That combination worked better than trying to artificially balance the dataset.

## Threshold Tuning

The default 0.5 threshold gave around 39% recall on defaulters. That's too low. I tested:

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.50 | 0.60 | 0.39 | 0.47 |
| 0.40 | 0.55 | 0.48 | 0.51 |
| 0.35 | 0.52 | 0.51 | 0.52 |
| 0.30 | 0.48 | 0.56 | 0.51 |
| 0.25 | 0.43 | 0.61 | 0.51 |

I chose **0.25**. Recall jumps to 61%, meaning we catch more than half of actual defaulters. Precision drops to 43%, so there are more false alarms — but that's a trade-off I'd rather make in a credit risk context. Missing a default is expensive; flagging a non-defaulter for review is cheap.

## Model Choice

I compared Logistic Regression, Decision Tree, and Random Forest. Random Forest performed best in terms of recall-precision balance after threshold tuning. But honestly, the model wasn't the focus — the feature engineering and threshold choices mattered more.

Hyperparameter tuning across `max_depth` and `min_samples_leaf` didn't change results dramatically. The best configuration was default settings with balanced class weights.

## Final Results

With threshold at 0.25 and balanced class weights:

```
              precision    recall  f1-score   support

           0       0.88      0.78      0.83      7052
           1       0.43      0.61      0.51      1948

    accuracy                           0.74      9000
```

Accuracy is ~74%, lower than baseline models — but that's on purpose. We're trading some accuracy for substantially better recall on the default class.

## Limitations

- The dataset is from 2005 Taiwan. Economic behavior and credit norms may differ elsewhere.
- Demographic features removed; including them might improve accuracy but raises fairness concerns.
- Threshold was optimized on test set; would need validation set for production use.
- No probability calibration applied — predicted probabilities may not reflect true likelihoods.

## Next Steps (If Continued)

- Cross-validation for threshold selection instead of single train-test split
- Explore gradient boosting (XGBoost/LightGBM) with built-in handling for imbalance
- Probability calibration for more interpretable scores
- Feature importance analysis to explain predictions

---

Built with scikit-learn. AI was used as a coding assistant, but the analytical decisions and trade-offs are my own.
