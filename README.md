# Thera-Bank-Loan-Propensity


Built an end-to-end binary classification pipeline on TheraBank's dataset to predict which liability customers are most likely to convert to personal loan holders. The model uses industry-standard retail credit risk methodology: quantile binning → Weight of Evidence (WoE) transformation → Information Value (IV) feature selection → Logistic Regression scoring.
Results: AUC 0.969 · KS 0.811 · Gini 0.938 · Accuracy 96.8%

Stack: Python · Pandas · Scikit-learn · Matplotlib · Custom WoE engine (from scratch)

