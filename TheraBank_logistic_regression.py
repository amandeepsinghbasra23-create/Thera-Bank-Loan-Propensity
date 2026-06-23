import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

#LOAD DATA
df = pd.read_csv('TheraBank.csv')
target = 'Personal Loan'

# Clean column names (strip trailing spaces if any)
df.columns = df.columns.str.strip()
df.drop(columns=['ID', 'ZIP Code'], inplace=True, errors='ignore')

#PREPROCESSING & BINNING FOR WoE
# Handle negative values across numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != target:
        df[col] = df[col].apply(lambda x: 0 if x < 0 else x)

X_binned = pd.DataFrame()
woe_maps = {}
iv_summary = {}
master_rows = []  # List to store data for the single summary table

print("Calculating WoE and IV for features...")

for col in df.columns:
    if col == target:
        continue
        
    s = df[col].copy()
    is_blank = s.isna() | (s.astype(str).str.strip() == '')
    
    if s[~is_blank].nunique() > 5:  # Continuous feature
        try:
            bins = pd.qcut(s[~is_blank], q=5, duplicates='drop')
            s_binned = bins.astype(str)
        except Exception:
            bins = pd.cut(s[~is_blank], bins=5)
            s_binned = bins.astype(str)
            
        woe_maps[col] = {
            'type': 'continuous', 
            'bins': pd.qcut(s.dropna(), q=5, duplicates='drop').cat.categories if 'bins' in locals() else pd.cut(s.dropna(), bins=5).cat.categories
        }
    else:  # Categorical/Discrete feature
        s_binned = s.astype(str)
        woe_maps[col] = {'type': 'categorical'}
        
    # Inject 'Blank' for actual nulls or empty spaces
    s_binned[is_blank] = 'Blank'
    X_binned[col] = s_binned

    # --- Calculate WoE and IV ---
    temp_df = pd.DataFrame({'Feature': X_binned[col], 'Target': df[target]})
    total_events = temp_df['Target'].sum()
    total_non_events = temp_df['Target'].count() - total_events
    
    stats = temp_df.groupby('Feature')['Target'].agg(Events='sum', Total='count')
    stats['Non_Events'] = stats['Total'] - stats['Events']
    
    # Adjust for zero counts to avoid log(0) errors
    stats['Events'] = stats['Events'].replace(0, 0.5)
    stats['Non_Events'] = stats['Non_Events'].replace(0, 0.5)
    
    stats['Dist_Events'] = stats['Events'] / total_events
    stats['Dist_Non_Events'] = stats['Non_Events'] / total_non_events
    
    stats['WoE'] = np.log(stats['Dist_Non_Events'] / stats['Dist_Events'])
    stats['IV'] = (stats['Dist_Non_Events'] - stats['Dist_Events']) * stats['WoE']
    
    woe_maps[col]['map'] = stats['WoE'].to_dict()
    total_feature_iv = stats['IV'].sum()
    iv_summary[col] = total_feature_iv

    for bin_name, row in stats.iterrows():
        master_rows.append({
            'Feature': col,
            'Bin/Category': bin_name,
            'No Loan (Non-Events)': int(row['Non_Events']),
            'Took Loan (Events)': int(row['Events']),
            'Total': int(row['Total']),
            'WoE Value': round(row['WoE'], 4),
            'Bin IV Contribution': round(row['IV'], 4),
            'Total Feature IV': round(total_feature_iv, 4)
        })

#3. DISPLAY THE UNIFIED BINS AND WOE MASTER TABLE 
master_table = pd.DataFrame(master_rows)
print("\n" + "="*95)
print("❖ MASTER BINS, WEIGHT OF EVIDENCE (WoE), AND INFORMATION VALUE (IV) MATRIX ❖")
print("="*95)
try:
    print(master_table.to_markdown(index=False, tablefmt="grid"))
except ImportError:
    print(master_table.to_string(index=False))

# 4. FEATURE SELECTION BASED ON IV > 0.1
selected_features = [feature for feature, iv in iv_summary.items() if iv > 0.1]

print("\n========== FEATURE SELECTION SUMMARY ==========")
for feature, iv in sorted(iv_summary.items(), key=lambda x: x[1], reverse=True):
    status = "SELECTED" if feature in selected_features else "DROPPED (IV <= 0.1)"
    print(f"{feature:<25} | Total IV: {iv:.4f} | Status: {status}")

if len(selected_features) == 0:
    raise ValueError("No features have an IV greater than 0.1! Adjust threshold or check data.")

#5. APPLY WoE TRANSFORMATION ONLY TO SELECTED FEATURES 
X_woe = pd.DataFrame()
for col in selected_features:
    X_woe[col] = X_binned[col].map(woe_maps[col]['map'])

y = df[target]

# 6. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_woe, y, test_size=0.2, random_state=42, stratify=y
)

#7. TRAIN LOGISTIC REGRESSION 
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

#8. EVALUATION METRICS 
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc         = roc_auc_score(y_test, y_prob)
gini        = 2 * auc - 1
fpr, tpr, _ = roc_curve(y_test, y_prob)
ks          = float(np.max(tpr - fpr))

print("\n========== MODEL PERFORMANCE (USING SELECTED FEATURE WoE) ==========")
print(f"AUC-ROC  : {auc:.4f}")
print(f"Gini     : {gini:.4f}")
print(f"KS Stat  : {ks:.4f}")
print(f"Accuracy : {(y_pred == y_test).mean():.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create DataFrame for plotting coefficients
coef_df = pd.DataFrame({
    'Feature'    : X_train.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

#9. GENERATE RESULTS PLOTS
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('TheraBank — Filtered WoE Logistic Regression Results', fontsize=14, fontweight='bold')

# Plot 1: ROC Curve
axes[0].plot(fpr, tpr, color='#4A90E2', lw=2, label=f'AUC = {auc:.4f}')
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Loan', 'Loan'])
disp.plot(ax=axes[1], colorbar=False, cmap='Greens')
axes[1].set_title('Confusion Matrix')

# Plot 3: Feature Coefficients (WoE Impact)
colors = ['#2ECC71' if v >= 0 else '#E74C3C' for v in coef_df['Coefficient']]
axes[2].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
axes[2].axvline(0, color='black', linewidth=0.8)
axes[2].set_xlabel('Weight of Coefficient Impact')
axes[2].set_title('WoE Feature Coefficients')
axes[2].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# 10. PREDICT ON NEW CUSTOMER 
def map_value_to_woe(col_name, val):
    meta = woe_maps[col_name]
    
    if pd.isna(val) or val == '':
        category = 'Blank'
    elif isinstance(val, (int, float)) and val < 0:
        val = 0
        
    if meta['type'] == 'continuous' and not (pd.isna(val) or val == 'Blank'):
        category = 'Blank'
        for interval in meta['bins']:
            if val in interval:
                category = str(interval)
                break
        if category == 'Blank':
            if val <= meta['bins'][0].left:
                category = str(meta['bins'][0])
            else:
                category = str(meta['bins'][-1])
    elif meta['type'] == 'categorical':
        category = str(val)
        
    return meta['map'].get(category, meta['map'].get('Blank', 0.0))

def get_age(age_text):
    while True:
        try:
            val = input(age_text).strip()
            if val == '': return np.nan
            answer = int(val)
            if answer < 18: print("  -> Error: Under 18 is too young.")
            elif answer > 100: print("  -> Error: Over 100 is too old.")
            else: return answer 
        except ValueError:
            print("  -> Error: Please type a valid number.")

def get_numeric(prompt_text):
    val = input(prompt_text).strip()
    return np.nan if val == '' else float(val)

def get_family(text_family):
    while True:
         val = input(text_family).strip()
         if val == '': return np.nan
         answer = int(val)
         if 1 <= answer <= 4: return answer
         else: print("-> Error: Choose between 1 and 4.")

def get_education(text_edu):
    while True:
        answer = input(text_edu).strip().lower()
        if answer == '': return np.nan
        if answer in ['higher school', 'school', 'highschool', 'high','+2']: return 1
        elif answer in ['secondary education', 'college', 'university', 'undergraduate', 'higher education', 'higher', 'higher edu', 'undergrad', 'uni']: return 2
        elif answer in ['masters', 'post grad', 'post graduation', 'graduate', 'postgrad']: return 3
        else: print("Error: Invalid entry.")

def get_yes_no(prompt_text):
    while True:
        answer = input(prompt_text).strip().lower()
        if answer == '': return np.nan
        if answer in ['yes', 'y']: return 1
        elif answer in ['no', 'n']: return 0
        else: print("  -> Error: Please type 'yes' or 'no'.")

print("\n--- Enter New Customer Details ---")
# Prompt inputs dynamically based on which features survived the filter
all_prompts = {
    'Age (in years)'       : lambda: get_age("What is your age? "),
    'Experience (in years)': lambda: get_numeric("What is your experience? "),
    'Income (in K/month)'  : lambda: get_numeric("What is your income (in K/Month)? "),
    'Family members'       : lambda: get_family("How many family members? (1-4): "),
    'CCAvg'                : lambda: get_numeric("What is your CCAvg? "),
    'Education'            : lambda: get_education("What is your education level? (school/university/masters): "),
    'Mortgage'             : lambda: get_yes_no("Do you have a mortgage? (yes/no): "),
    'Securities Account'   : lambda: get_yes_no("Security Account? (yes/no): "),
    'CD Account'           : lambda: get_yes_no("CD Account? (yes/no): "),
    'Online'               : lambda: get_yes_no("Online? (yes/no): "),
    'CreditCard'           : lambda: get_yes_no("Credit card? (yes/no): ")
}

raw_inputs = {}
for col in selected_features:
    if col in all_prompts:
        raw_inputs[col] = all_prompts[col]()

# Convert only the collected inputs to WoE space
new_cust_woe = {}
for col in raw_inputs.keys():
    new_cust_woe[col] = map_value_to_woe(col, raw_inputs[col])

new_customer_df = pd.DataFrame([new_cust_woe])[X_train.columns]

# Generate probability prediction
prob = model.predict_proba(new_customer_df)[0][1]
pred = model.predict(new_customer_df)[0]

print("\n========== INDIVIDUAL PREDICTION RESULT ==========")
print(f"Probability of taking a personal loan : {prob:.4f} ({prob*100:.1f}%)")
print(f"Prediction (0=No Loan, 1=Loan)        : {pred}")
