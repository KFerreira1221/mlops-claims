\## Fraud Detection Model (Classification)



\*\*Goal:\*\* predict whether a claim is fraudulent (`fraud\_reported`) using policy + customer + incident details.



\### Target \& Leakage Control

To reduce information leakage, the model drops claim payout fields that can indirectly reveal the label:

\- `total\_claim\_amount`, `injury\_claim`, `property\_claim`, `vehicle\_claim`



This forces the classifier to learn fraud patterns from contextual and behavioral signals rather than “cheating” from outcome fields.



\### Model

\- Algorithm: \*\*XGBoost (XGBClassifier)\*\*

\- Preprocessing:

&nbsp; - Numeric: median imputation

&nbsp; - Categorical: most-frequent imputation + one-hot encoding

\- Imbalance handling: `scale\_pos\_weight` computed from training class balance



\### Evaluation

Fraud is typically imbalanced, so we report more than accuracy:

\- \*\*AUC\*\* (ranking quality)

\- \*\*Precision / Recall / F1\*\* (tradeoffs between false alarms and missed fraud)

\- \*\*Confusion Matrix\*\* (TN/FP/FN/TP)



Metrics are saved to:

\- `reports/fraud\_metrics.json` (includes AUC + confusion matrix + threshold sweep)



\### Threshold Tuning

Fraud detection often uses an adjustable decision threshold:

\- Lower threshold → higher recall (catch more fraud) but more false positives

\- Higher threshold → higher precision (fewer false positives) but more missed fraud



This project logs a threshold sweep at: \*\*0.30, 0.40, 0.50, 0.60, 0.70\*\* in `reports/fraud\_metrics.json` to support cost-sensitive decision-making.



\### Interpretability

Feature importance is exported to:

\- `reports/fraud\_feature\_importance.csv`



This helps identify which policy/incident features most influence fraud risk and supports stakeholder-facing explanations.



