"""
Train all models and save them to disk for deployment.
Run this ONCE locally to generate the saved_models/ directory.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os
import json

OUTPUT_DIR = "saved_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv("creditcard.csv")

print("Preprocessing...")
scaler = RobustScaler()
df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df = df.drop(['Time', 'Amount'], axis=1)

X = df.drop('Class', axis=1)
y = df['Class']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# --- Train all models ---
models = {}

print("Training Baseline LR...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
models['Baseline LR'] = lr

print("Training Baseline RF...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
models['Baseline RF'] = rf

print("Training SMOTE LR...")
lr_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_smote.fit(X_train_smote, y_train_smote)
models['SMOTE LR'] = lr_smote

print("Training Cost-Sensitive RF...")
rf_cost = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=42)
rf_cost.fit(X_train, y_train)
models['Cost-Sensitive RF'] = rf_cost

print("Training XGBoost...")
fraud_count = y_train.sum()
legit_count = len(y_train) - fraud_count
xgb = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=legit_count / fraud_count,
    use_label_encoder=False, eval_metric='aucpr',
    n_jobs=-1, random_state=42
)
xgb.fit(X_train, y_train)
models['XGBoost'] = xgb

# --- Evaluate, save predictions, and print ---
print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)
all_probs = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    all_probs[name] = y_prob
    print(f"\n{name}:")
    print(f"  Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"  F1:        {f1_score(y_test, y_pred):.3f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_prob):.3f}")

# --- Save everything ---
print("\nSaving models...")
for name, model in models.items():
    safe_name = name.lower().replace(' ', '_').replace('-', '_')
    joblib.dump(model, os.path.join(OUTPUT_DIR, f"{safe_name}.joblib"))
    print(f"  Saved {name} → {safe_name}.joblib")

joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
print("  Saved scaler")

# Save FULL test set y_test and y_probs (for accurate metric computation on cloud)
np.save(os.path.join(OUTPUT_DIR, "y_test_full.npy"), y_test.values)
probs_dict = {name: prob.tolist() for name, prob in all_probs.items()}
with open(os.path.join(OUTPUT_DIR, "y_probs_full.json"), 'w') as f:
    json.dump(probs_dict, f)
print(f"  Saved full test predictions (y_test: {len(y_test)}, probs for {len(all_probs)} models)")

# Save SMALL test sample (for Live Predictor feature loading only)
test_full = pd.DataFrame(X_test, columns=feature_names)
test_full['Class'] = y_test.values
fraud_rows = test_full[test_full['Class'] == 1]
legit_rows = test_full[test_full['Class'] == 0].sample(n=2000, random_state=42)
test_sample = pd.concat([fraud_rows, legit_rows]).sample(frac=1, random_state=42)
test_sample.to_csv(os.path.join(OUTPUT_DIR, "test_sample.csv"), index=False)
print(f"  Saved test_sample.csv ({len(test_sample)} rows for Live Predictor)")

# Save feature names
with open(os.path.join(OUTPUT_DIR, "feature_names.json"), 'w') as f:
    json.dump(feature_names, f)
print("  Saved feature_names.json")

print(f"\n✅ All models saved to {OUTPUT_DIR}/")
print("You can now deploy without needing creditcard.csv!")
