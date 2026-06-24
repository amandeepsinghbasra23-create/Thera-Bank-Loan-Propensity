# Loan-Propensity-WOE-IVs


🏦 TheraBank Personal Loan Propensity Modeling 📈
📌 Project Overview
TheraBank, a growing financial institution, aims to expand its asset base by converting existing liability customers (depositors) into personal loan customers while retaining their deposit business. This project involves analyzing the results of a previous marketing campaign to build predictive classification models that identify the customers most likely to accept a loan offer, thereby maximizing campaign ROI.
📊 The Data Challenge
The dataset comprises demographic and behavioral data for 5,000 customers. A significant challenge in this analysis is the highly imbalanced target variable, with only 480 customers (9.6%) accepting the loan in the historical campaign.
🛠️ Methodology & Technical Approach
1. Data Cleaning & Exploratory Data Analysis (EDA)
Data Integrity: Corrected structural anomalies, such as setting negative values in the Experience column to zero, and isolated missing values in the Family members column to prevent data loss.
Bivariate Analysis: EDA revealed that high-income earners (median ~$140K) and customers with higher education levels (Graduates/Professionals) were significantly more likely to accept a loan offer.
Multicollinearity: Detected a near-perfect positive correlation (r=0.99) between Age and Experience, highlighting the need for careful feature selection during modeling.
2. Feature Engineering (WoE & IV)
Conducted Weight of Evidence (WoE) and Information Value (IV) analysis to rigorously evaluate the predictive strength of all features.
Findings: Income and CCAvg (Credit Card Average Spend) emerged as exceptionally strong predictors. Conversely, demographic features like Age and Experience showed almost zero predictive power for this specific financial product.
3. Machine Learning Pipelines
To provide a robust solution, I developed two separate classification pipelines to compare performance and interpretability:
Model A: Logistic Regression with SMOTE (Focus on Interpretability & Recall)
Constructed a robust preprocessing pipeline using ColumnTransformer (StandardScaler for numerical data, OneHotEncoder for categorical data).
Addressed the 9.6% class imbalance by applying SMOTE (Synthetic Minority Over-sampling Technique) to the training data.
Results: Achieved 95.8% Accuracy and an F1-score of 0.81. The model successfully balanced high precision (0.87) with strong recall (0.76), providing interpretable coefficients for business stakeholders.
Model B: Gaussian Naive Bayes (Probabilistic Baseline)
Implemented a GaussianNB pipeline to calculate the conditional probability of loan acceptance based on the engineered feature vector, assuming feature independence.
🚀 Business Value
The deployed models empower TheraBank to transition away from inefficient mass marketing. By applying the Logistic Regression model to their current customer base, the marketing team can generate a scored, prioritized list of prospects. Targeting high-probability profiles—specifically high-income professionals who actively use credit cards—will drastically improve campaign conversion rates and reduce customer acquisition costs.
💻 Tech Stack
Language: Python
Libraries: Pandas, NumPy, Scikit-Learn, Imbalanced-Learn
Algorithms: Logistic Regression, Gaussian Naive Bayes
Techniques: SMOTE, WoE/IV Analysis, Scikit-Learn Pipelines (ColumnTransformer)
Visualization: Matplotlib, Seaborn

