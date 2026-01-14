# Credit Card Default Prediction

A practical machine learning project for predicting credit card payment defaults. The focus here isn't on chasing high accuracy numbers, but on making decisions that actually matter in a credit risk context—specifically, catching defaults before they happen.

## The Problem

The UCI Credit Card dataset contains ~30,000 customer records with payment history, bill amounts, and demographic information. About 22% of customers defaulted on their next payment. That imbalance is the first thing that shapes everything else.

### Why Accuracy Doesn't Work Here

With a 78/22 class split, a model that predicts "no default" for everyone gets 78% accuracy. Useless. In credit risk, missing actual defaults is far more expensive than flagging a few good customers for extra review. So **recall on the minority class (defaults) became the primary metric**, with precision as the constraint to keep false alarms manageable.

## What the Data Actually Shows

The EDA wasn't about generating pretty charts—it was about understanding what features carry signal and which ones just add noise.

### High Correlation Among Temporal Features

The correlation matrix revealed something important: the six monthly bill amounts (BILL_AMT1-6) are highly correlated with each other (0.9+). Same pattern for the payment status columns (PAY_0, PAY_2-6). Including all of them individually would be redundant and could hurt generalization.

### Payment Status Matters Most

Looking at distributions across default vs non-default groups, repayment status (how many months delayed) showed the clearest separation. Bill amounts and age showed some pattern but weaker. Credit limit had modest predictive value.

## Feature Engineering Choices

Instead of throwing PCA at the data blindly, I aggregated the temporally correlated features based on what they actually represent:

**What I created:**
- `max_delay` / `avg_delay` — aggregate repayment status across 6 months
- `max_bill` / `avg_bill` / `bill_trend` — bill amount patterns
- `max_pay_amt` / `avg_pay_amt` — payment behavior summary

**What I dropped:**
- Individual monthly columns (PAY_2-6, BILL_AMT2-6, PAY_AMT2-6) — redundant after aggregation
- ID, SEX, EDUCATION, MARRIAGE, AGE — either non-predictive or not consistently useful

The goal was reducing dimensionality while preserving behavioral signal, not just mathematical variance.

## Handling Class Imbalance

I tried several approaches:

1. **Baseline Random Forest** — high accuracy, terrible recall (~39%)
2. **Random Forest with `class_weight='balanced'`** — sklearn automatically upweights the minority class during training
3. **Threshold tuning** — instead of using 0.5 probability cutoff, evaluated 0.5, 0.4, 0.35, 0.3, 0.25

The class weights helped, but the real improvement came from adjusting the decision threshold.

## Threshold Selection

With probability predictions from Random Forest, I tested different cutoffs:

| Threshold | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| 0.50 | 0.60 | 0.39 | 0.47 |
| 0.40 | 0.55 | 0.48 | 0.51 |
| 0.35 | 0.52 | 0.51 | 0.52 |
| 0.30 | 0.48 | 0.56 | 0.51 |
| 0.25 | 0.43 | 0.61 | 0.51 |

**I chose 0.25** as the final threshold. Recall jumps to ~61%, meaning we catch roughly 6 out of 10 defaults. Precision drops to ~43%, so there are more false positives—but in a credit context, reviewing extra applications is far cheaper than absorbing defaults.

This is a business decision, not a purely technical one. Different cost ratios would lead to different thresholds.

## Model Comparison

Tested three classifiers:

- **Logistic Regression**: 81% accuracy, 23% recall on defaults. Too conservative.
- **Decision Tree**: 72% accuracy, 40% recall. Unstable.
- **Random Forest**: With tuning, 74% accuracy, 61% recall. Better trade-off.

Random Forest won not because it's sophisticated, but because its ensemble averaging reduces variance and the probability outputs are well-calibrated enough for threshold tuning.

## Final Results

Using Random Forest with balanced class weights, threshold 0.25:

```
              precision    recall  f1-score   support

           0       0.88      0.78      0.83      7052
           1       0.43      0.61      0.51      1948

    accuracy                           0.74      9000
```

- **61% of actual defaults caught** (recall)
- **43% of flagged accounts are true defaults** (precision)
- **74% overall accuracy** (down from 82% at default threshold, but actually useful)

## Limitations

- The model relies heavily on repayment history—new customers with no history would need a different approach
- Threshold choice assumes specific cost trade-offs that may not match every business
- No external validation; performance on truly new data is unknown
- Hyperparameter tuning was limited (depth and leaf constraints); more extensive search might help
- Feature engineering was manual; automated approaches (gradient boosting, neural nets) might find interactions I missed

## What I'd Do Next

- Test on held-out time periods (true temporal validation)
- Calibrate the probability outputs more carefully
- Estimate actual costs of false positives vs false negatives to set threshold more rigorously
- Try gradient boosting (XGBoost/LightGBM) for comparison
- Add SHAP values for individual prediction explanations

---

*This project used AI assistance for code generation and debugging, but the design decisions, feature engineering logic, and threshold selection reflect my own reasoning about the problem.*