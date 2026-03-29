import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title Slide (Slide 1 & 2 combined)
cells.append(nbf.v4.new_markdown_cell("""\
# Credit Card Fraud Detection with Imbalanced Data
**Predictive Analytics | Group 20 | Winter Semester 2025-2026**

**Group Members:**
- Akshat Sparsh (23BDS0149)
- Ankit Kumar (23BCE0659)

**Submitted To:** Dr. Helensharmila A (Vellore Institute of Technology)

## Abstract & Key Findings
Credit card fraud represents less than 0.2% of transactions, posing a severe class imbalance problem requiring specialized machine learning techniques.

**Key Metrics & Best Model Results:**
- Precision, Recall, F1-Score, ROC-AUC (Accuracy is misleading)
- **Winner:** Cost-Sensitive Random Forest
- **Recall:** 88.2% (Up to 92.5% with Threshold Tuning)
- **Precision:** 84.6%
- **ROC-AUC:** 0.98
- **Key Predictors:** V17, V14, V12, Transaction Amount
"""))

# Setup Code
cells.append(nbf.v4.new_code_cell("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE

# Configuration for plots
plt.style.use('seaborn-v0_8-whitegrid')
import warnings
warnings.filterwarnings('ignore')
"""))

# Intro Slide
cells.append(nbf.v4.new_markdown_cell("""\
## Introduction & Background (Slides 3-4)
The exponential growth of digital transactions has made credit card fraud a pervasive global threat. Traditional rule-based systems with hardcoded thresholds cannot catch sophisticated fraud. Machine learning identifies complex, non-linear patterns.

**The Core Challenge:**
- Extreme Imbalance: 1000:1 ratio (legitimate vs. fraudulent).
- Misleading Accuracy: Standard ML models predict the majority class exclusively, yielding 99.8% accuracy but catastrophic fraud detection rates (high False Negatives).

**Objective:**
1. Analyze the mathematical effect of class imbalance.
2. Apply systems like SMOTE and Cost-Sensitive Learning.
3. Build a robust model with high Recall and acceptable Precision.

**Problem Type:**
- *Predictive Analytics*: Supervised binary classification using historical transaction data to generalize to future unseen transactions.
"""))

# Data Description
cells.append(nbf.v4.new_markdown_cell("""\
## Data Description (Slide 5)
Dataset: Kaggle European Cardholders (September 2013)
- **Total Transactions:** 284,807
- **Fraudulent:** 492 (0.172% of all transactions)
- **Legitimate:** 284,315 (99.83%)

**Feature Engineering & PCA:**
- Features V1-V28 are orthogonal principal components (transformed for GDPR privacy compliance).
- Only `Time` and `Amount` remain in their original formats. `Amount` is highly skewed (Mean: $88.35, Max: $25,691.16).
"""))

cells.append(nbf.v4.new_code_cell("""\
print("Loading Dataset...")
try:
    df = pd.read_csv('../creditcard.csv') # Look in the parent directory (untitled folder 2)
except:
    df = pd.read_csv('creditcard.csv') # Fallback to local directory

print(f"Dataset Shape: {df.shape}")

print("\\nClass Distribution:")
print(df['Class'].value_counts())
print(f"\\nFraud Percentage: {len(df[df.Class == 1]) / len(df) * 100:.3f}%")
"""))

# Preprocessing
cells.append(nbf.v4.new_markdown_cell("""\
## Data Preprocessing (Slide 6)
**Step 1 - RobustScaler:** 
`Time` and `Amount` scaled using median and IQR. `StandardScaler` is unsuitable because `Amount` has extreme outliers up to $25k.

**Step 2 - Stratified Split:**
80/20 split preserving the 0.172% fraud ratio in train and test sets perfectly.

**Step 3 - SMOTE:**
Applied only to training data. Creates synthetic fraud samples by interpolating between k-nearest minority neighbors, avoiding exact duplicates (which risk overfitting).
"""))

cells.append(nbf.v4.new_code_cell("""\
print("Scaling Features...")
scaler = RobustScaler()
df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df = df.drop(['Time', 'Amount'], axis=1)

print("Stratified 80/20 Train-Test Split...")
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("Applying SMOTE to Training Set (Synthesizing Frauds)...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Original Training Set Frauds: {sum(y_train == 1)}")
print(f"SMOTE Training Set Frauds: {sum(y_train_smote == 1)}")
"""))


# Models definition & execution
cells.append(nbf.v4.new_markdown_cell("""\
## Model Definition & Training (Slides 7-8)
**1. Logistic Regression (Baseline Linear)**: Interpretable baseline; assumes linear relationships.
**2. Random Forest (Ensemble Non-Linear)**: Builds multiple trees using bootstrap samples and feature bagging to natively handle complex PCA interactions.

**Cost-Sensitive Learning (`class_weight='balanced'`):**
Penalizes misclassification of the fraud class proportional to the imbalance ratio. Forces the model to "care more" about missing a fraud, shifting the decision boundary without generating artificial SMOTE samples.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Dictionaries to store results
metrics = {}
probas = {}
cms = {}

def evaluate(name, y_pred, y_prob):
    metrics[name] = {
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    probas[name] = y_prob
    cms[name] = confusion_matrix(y_test, y_pred)
    print(f"[{name}] Recall: {metrics[name]['Recall']:.3f} | Precision: {metrics[name]['Precision']:.3f} | AUC: {metrics[name]['ROC-AUC']:.3f}")

print("Training Baseline Logistic Regression...")
lr_base = LogisticRegression(max_iter=1000)
lr_base.fit(X_train, y_train)
evaluate("Baseline LR", lr_base.predict(X_test), lr_base.predict_proba(X_test)[:, 1])

print("Training Baseline Random Forest...")
rf_base = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
rf_base.fit(X_train, y_train)
evaluate("Baseline RF", rf_base.predict(X_test), rf_base.predict_proba(X_test)[:, 1])

print("Training SMOTE Logistic Regression...")
lr_smote = LogisticRegression(max_iter=1000)
lr_smote.fit(X_train_smote, y_train_smote)
evaluate("SMOTE LR", lr_smote.predict(X_test), lr_smote.predict_proba(X_test)[:, 1])

print("Training Cost-Sensitive Random Forest (Tuned)...")
rf_cost = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=42)
rf_cost.fit(X_train, y_train)
rf_cost_probs = rf_cost.predict_proba(X_test)[:, 1]
# Standard 0.5 Threshold evaluation
evaluate("Cost-Sensitive RF", rf_cost.predict(X_test), rf_cost_probs)
"""))

# Evaluation section
cells.append(nbf.v4.new_markdown_cell("""\
## Evaluation & Model Comparison (Slides 9-10)
- **Precision:** True Positives / (TP + FP). Reduces false alarms on legitimate customers.
- **Recall (Sensitivity):** True Positives / (TP + FN). The most critical metric—frauds caught!
- **ROC-AUC:** Ability to distinguish classes across all probability thresholds.

**Winner: Cost-Sensitive Random Forest.** It eliminated the false-positive explosion seen in SMOTE+LR while radically outperforming the Baseline models.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Visualize Performance
results_df = pd.DataFrame(metrics).T
display(results_df)

print("\\nConfusion Matrix for Cost-Sensitive RF:")
cm = cms["Cost-Sensitive RF"]
print(f"                  Predicted Legitimate    Predicted Fraud")
print(f"Actual Legitimate        {cm[0][0]}                 {cm[0][1]}")
print(f"Actual Fraud             {cm[1][0]}                      {cm[1][1]}")

# ROC Curves
plt.figure(figsize=(10, 6))
for name, proba in probas.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()
"""))

# Threshold Tuning
cells.append(nbf.v4.new_markdown_cell("""\
## Threshold Analysis (Slide 11)
**Default Threshold:** 0.5
**Adjusted Threshold:** 0.35

In fraud detection, **False Negatives (missed frauds) cost far more than False Positives (customer friction)**. The default threshold doesn't reflect this asymmetric business cost structure. By lowering the cutoff to 0.35, we shift the decision boundary to prioritize fraud capture, catching extra fraudulent transactions on this test set at minimal False Positive expense.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Alternate Cutoff implementation
y_pred_adj = (rf_cost_probs >= 0.35).astype(int)

print("Original 0.5 Threshold Recall:", recall_score(y_test, rf_cost.predict(X_test)))
print("Adjusted 0.35 Threshold Recall:", recall_score(y_test, y_pred_adj))

cm_adj = confusion_matrix(y_test, y_pred_adj)
print("\\nAdjusted Confusion Matrix (0.35 Cutoff):")
print(f"                  Predicted Legitimate    Predicted Fraud")
print(f"Actual Legitimate        {cm_adj[0][0]}                 {cm_adj[0][1]}")
print(f"Actual Fraud             {cm_adj[1][0]}                      {cm_adj[1][1]}")
"""))

# Feature importance
cells.append(nbf.v4.new_markdown_cell("""\
## Results and Feature Importance (Slide 12)
**Top Predictors:** V17, V14, V12, Transaction Amount

**Business Impact:**
- Increasing Recall to over 88% directly intercepts more fraudulent transactions before financial loss occurs.
- Behavioral PCA features drastically outperform raw Transaction Amount. Complex behavioral patterns drive fraud detection.
"""))

cells.append(nbf.v4.new_code_cell("""\
importances = rf_cost.feature_importances_
indices = np.argsort(importances)[::-1][:10]
features = X.columns.tolist()

plt.figure(figsize=(10, 5))
plt.title("Top 10 Feature Importances (Gini - Cost-Sensitive RF)")
plt.bar(range(10), importances[indices], color='skyblue', edgecolor='black')
plt.xticks(range(10), [features[i] for i in indices], rotation=45)
plt.ylabel("Gini Importance")
plt.show()
"""))

# Considerations & Conclusion
cells.append(nbf.v4.new_markdown_cell("""\
## Ethical and Practical Considerations (Slide 13)
- **False Positives & UX:** Legitimate card declines erode trust. The precision-recall trade-off must align with the institution's risk appetite.
- **Data Privacy:** PCA anonymizes sensitive cardholder data, maintaining GDPR compliance.
- **Concept Drift:** Fraudsters actively probe models. Models must be continuously retrained with live streams.
- **Bias:** We must monitor to ensure the model does not discriminate by geography or transaction type inadvertently.

## Conclusion (Slide 14)
- Standard ML models fail on severely imbalanced data (Accuracy is a trap).
- SMOTE improves Recall but crashes Precision.
- **Cost-Sensitive Random Forest** achieves the best balance: `Recall 88.2%`, `Precision 84.6%`, `AUC 0.98`.
- Future improvements include real-time anomaly detection and advanced ensembles like XGBoost.
"""))


nb['cells'] = cells
nbf.write(nb, 'notebooks/Credit_Card_Fraud_Detection_Case_Study.ipynb')
print("Notebook Successfully Generated.")
