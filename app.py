import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero-header h1 {
        color: #fff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .hero-header p {
        color: #a8a8d0;
        font-size: 1rem;
        margin: 0;
    }

    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-fraud {
        background: rgba(255, 75, 75, 0.15);
        color: #ff4b4b;
        border: 1px solid rgba(255, 75, 75, 0.3);
    }
    .status-legit {
        background: rgba(0, 210, 140, 0.15);
        color: #00d28c;
        border: 1px solid rgba(0, 210, 140, 0.3);
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 8px;
        padding: 10px 20px;
        color: #ccc;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3a7bd5, #00d2ff) !important;
        color: white !important;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading & Model Training (cached) ───
@st.cache_data
def load_and_train():
    import os, joblib, json

    # --- Mode 1: Load pre-trained models (for cloud deployment) ---
    if os.path.exists('saved_models') and not os.path.exists('creditcard.csv') and not os.path.exists('data/creditcard.csv'):
        models = {}
        model_files = {
            'Baseline LR': 'baseline_lr.joblib',
            'Baseline RF': 'baseline_rf.joblib',
            'SMOTE LR': 'smote_lr.joblib',
            'Cost-Sensitive RF': 'cost_sensitive_rf.joblib',
            'XGBoost': 'xgboost.joblib'
        }
        for name, fname in model_files.items():
            path = os.path.join('saved_models', fname)
            if os.path.exists(path):
                models[name] = joblib.load(path)

        with open('saved_models/feature_names.json') as f:
            feature_names = json.load(f)

        # Load FULL test set predictions for accurate metrics
        y_test_full = np.load('saved_models/y_test_full.npy')
        with open('saved_models/y_probs_full.json') as f:
            probs_full = json.load(f)

        y_test = pd.Series(y_test_full)
        results = {}
        for name in models:
            y_prob = np.array(probs_full[name])
            y_pred = (y_prob >= 0.5).astype(int)
            results[name] = {
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_prob),
                'y_prob': y_prob,
                'y_pred': y_pred
            }

        # Load small test sample for Live Predictor only
        test_df = pd.read_csv('saved_models/test_sample.csv')
        X_test = test_df.drop('Class', axis=1).values

        X_dummy = pd.DataFrame(columns=feature_names)
        return test_df, X_dummy, X_test, y_test, models, results

    # --- Mode 2: Train from scratch (local with CSV) ---
    try:
        df = pd.read_csv('creditcard.csv')
    except:
        df = pd.read_csv('data/creditcard.csv')

    scaler = RobustScaler()
    df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df_processed = df.drop(['Time', 'Amount'], axis=1)

    X = df_processed.drop('Class', axis=1)
    y = df_processed['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    models = {}

    lr_base = LogisticRegression(max_iter=1000)
    lr_base.fit(X_train, y_train)
    models['Baseline LR'] = lr_base

    rf_base = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf_base.fit(X_train, y_train)
    models['Baseline RF'] = rf_base

    lr_smote = LogisticRegression(max_iter=1000)
    lr_smote.fit(X_train_smote, y_train_smote)
    models['SMOTE LR'] = lr_smote

    rf_cost = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=42)
    rf_cost.fit(X_train, y_train)
    models['Cost-Sensitive RF'] = rf_cost

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

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob),
            'y_prob': y_prob,
            'y_pred': y_pred
        }

    return df, X, X_test, y_test, models, results

# ─── Load Everything ───
with st.spinner("🔄 Training ML Models on 284,807 transactions..."):
    df, X, X_test, y_test, models, results = load_and_train()

# ─── HERO HEADER ───
st.markdown("""
<div class="hero-header">
    <h1>🛡️ Credit Card Fraud Detection</h1>
    <p>Predictive Analytics &bull; Group 20 &bull; Akshat Sparsh (23BDS0149) &bull; Ankit Kumar (23BCE0659)</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.markdown("---")
    selected_model = st.selectbox("🤖 Select Model", list(models.keys()), index=3)
    threshold = st.slider("🎯 Decision Threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Lower = catch more fraud (higher recall), Higher = fewer false alarms (higher precision)")
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.metric("Total Transactions", "284,807")
    st.metric("Fraudulent", "492")
    st.metric("Fraud Ratio", "0.173%")

# ─── Threshold-adjusted metrics ───
model = models[selected_model]
y_prob = model.predict_proba(X_test)[:, 1]
y_pred_adj = (y_prob >= threshold).astype(int)

prec = precision_score(y_test, y_pred_adj, zero_division=0)
rec = recall_score(y_test, y_pred_adj)
f1 = f1_score(y_test, y_pred_adj, zero_division=0)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred_adj)

# ─── KPI Cards ───
c1, c2, c3, c4 = st.columns(4)
for col, label, val in zip([c1, c2, c3, c4],
                           ["Precision", "Recall", "F1-Score", "ROC-AUC"],
                           [prec, rec, f1, auc]):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val:.1%}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ───
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 ROC Curves", "🔢 Confusion Matrix", "🏆 Feature Importance", "🔍 Live Predictor", "🧠 SHAP Explainability"])

with tab1:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    colors = ['#3a7bd5', '#ff6b6b', '#00d2ff', '#ffd93d', '#a855f7']
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        linewidth = 3 if name == selected_model else 1.5
        alpha = 1.0 if name == selected_model else 0.4
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['ROC-AUC']:.3f})", color=color, lw=linewidth, alpha=alpha)
    ax.plot([0, 1], [0, 1], 'w--', alpha=0.3, label="Random Chance")
    ax.set_xlabel('False Positive Rate', color='white', fontsize=11)
    ax.set_ylabel('True Positive Rate', color='white', fontsize=11)
    ax.set_title('ROC Curve Comparison', color='white', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333')
    st.pyplot(fig)

with tab2:
    col1, col2 = st.columns([1, 1])
    with col1:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('#0e1117')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                    xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'],
                    annot_kws={"size": 16, "weight": "bold"})
        ax2.set_xlabel('Predicted', color='white', fontsize=11)
        ax2.set_ylabel('Actual', color='white', fontsize=11)
        ax2.set_title(f'Confusion Matrix (Threshold={threshold})', color='white', fontsize=13, fontweight='bold')
        ax2.tick_params(colors='white')
        st.pyplot(fig2)
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"**True Positives (Frauds Caught):** `{tp}`")
        st.markdown(f"**False Negatives (Frauds Missed):** `{fn}`")
        st.markdown(f"**False Positives (False Alarms):** `{fp}`")
        st.markdown(f"**True Negatives (Correct Legit):** `{tn}`")
        st.markdown("---")
        total_fraud = tp + fn
        st.markdown(f"**🎯 {tp} out of {total_fraud} fraudulent transactions detected** ({tp/total_fraud*100:.1f}%)")
        if fn > 0:
            st.warning(f"⚠️ {fn} fraudulent transactions were missed. Consider lowering the threshold.")
        else:
            st.success("✅ All fraudulent transactions detected!")

with tab3:
    rf_model = models['Cost-Sensitive RF']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    features = X.columns.tolist()

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fig3.patch.set_facecolor('#0e1117')
    ax3.set_facecolor('#0e1117')
    bars = ax3.barh(range(15), importances[indices][::-1],
                    color=plt.cm.cool(np.linspace(0.2, 0.8, 15)), edgecolor='none')
    ax3.set_yticks(range(15))
    ax3.set_yticklabels([features[i] for i in indices][::-1], color='white', fontsize=10)
    ax3.set_xlabel('Gini Importance', color='white', fontsize=11)
    ax3.set_title('Top 15 Feature Importances (Cost-Sensitive RF)', color='white', fontsize=14, fontweight='bold')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values():
        spine.set_color('#333')
    st.pyplot(fig3)

with tab4:
    st.markdown("### 🔍 Predict a Single Transaction")
    st.markdown("Test the model on **real transactions** from the dataset, or tweak values manually.")

    # Load real examples from the dataset
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    y_test_arr = y_test.values

    fraud_indices = np.where(y_test_arr == 1)[0]
    legit_indices = np.where(y_test_arr == 0)[0]

    top_features = ['V17', 'V14', 'V12', 'V10', 'V16', 'V11', 'V4', 'V3', 'Amount_scaled']

    # Initialize session state with a fraud example on first load
    if 'sample_loaded' not in st.session_state:
        first_sample = X_test_df.iloc[fraud_indices[0]]
        st.session_state.sample_label = int(y_test_arr[fraud_indices[0]])
        for feat in top_features:
            st.session_state[f"feat_{feat}"] = float(first_sample[feat])
        # Store full sample for non-top features
        st.session_state.full_sample = first_sample.to_dict()
        st.session_state.sample_loaded = True

    def load_sample(category):
        if category == "fraud":
            idx = fraud_indices[np.random.randint(len(fraud_indices))]
        elif category == "legit":
            idx = legit_indices[np.random.randint(len(legit_indices))]
        else:
            idx = np.random.randint(len(X_test_df))
        sample = X_test_df.iloc[idx]
        st.session_state.sample_label = int(y_test_arr[idx])
        for feat in top_features:
            st.session_state[f"feat_{feat}"] = float(sample[feat])
        st.session_state.full_sample = sample.to_dict()

    st.markdown("#### Quick Load: Real Transactions from Dataset")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        st.button("🚨 Load a Real Fraud Transaction", use_container_width=True,
                  on_click=load_sample, args=("fraud",))
    with preset_col2:
        st.button("✅ Load a Real Legitimate Transaction", use_container_width=True,
                  on_click=load_sample, args=("legit",))
    with preset_col3:
        st.button("🎲 Load Random Transaction", use_container_width=True,
                  on_click=load_sample, args=("random",))

    actual_label = st.session_state.sample_label

    st.markdown("---")
    st.markdown("#### Transaction Feature Values")
    st.caption(f"Source: {'🚨 Known Fraud' if actual_label == 1 else '✅ Known Legitimate'} transaction from test set")

    col_a, col_b, col_c = st.columns(3)
    cols = [col_a, col_b, col_c]
    for i, feat in enumerate(top_features):
        with cols[i % 3]:
            st.number_input(f"{feat}", format="%.4f", key=f"feat_{feat}")

    # Build the full feature vector using stored sample + edited top features
    full_sample = st.session_state.full_sample.copy()
    for feat in top_features:
        full_sample[feat] = st.session_state[f"feat_{feat}"]
    input_array = np.array([full_sample[col] for col in X.columns]).reshape(1, -1)

    if st.button("🚀 Run Prediction", use_container_width=True):
        prob = model.predict_proba(input_array)[0][1]
        prediction = "FRAUD" if prob >= threshold else "LEGITIMATE"

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            if prob >= threshold:
                st.markdown(f"""
                <div style="text-align:center; padding: 2rem;">
                    <span class="status-badge status-fraud">🚨 FRAUD DETECTED</span>
                    <h2 style="color: #ff4b4b; margin-top: 1rem;">Fraud Probability: {prob:.1%}</h2>
                    <p style="color: #888;">Exceeds the {threshold:.0%} threshold → <strong>BLOCKED</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align:center; padding: 2rem;">
                    <span class="status-badge status-legit">✅ LEGITIMATE</span>
                    <h2 style="color: #00d28c; margin-top: 1rem;">Fraud Probability: {prob:.1%}</h2>
                    <p style="color: #888;">Below the {threshold:.0%} threshold → <strong>APPROVED</strong></p>
                </div>
                """, unsafe_allow_html=True)
        with res_col2:
            actual_text = "🚨 Actually Fraud" if actual_label == 1 else "✅ Actually Legitimate"
            correct = (prediction == "FRAUD" and actual_label == 1) or (prediction == "LEGITIMATE" and actual_label == 0)
            st.markdown(f"""
            <div style="text-align:center; padding: 2rem;">
                <h4 style="color: #888;">Ground Truth</h4>
                <h2 style="color: {'#ff4b4b' if actual_label == 1 else '#00d28c'};">{actual_text}</h2>
                <p style="color: {'#00d28c' if correct else '#ff4b4b'}; font-weight: bold;">
                    {'✅ Model is CORRECT!' if correct else '❌ Model made an error'}
                </p>
            </div>
            """, unsafe_allow_html=True)

with tab5:
    st.markdown("### 🧠 SHAP: Why Did the Model Make This Decision?")
    st.markdown("SHAP (SHapley Additive exPlanations) shows exactly **which features** pushed the prediction toward fraud or legitimate for a specific transaction.")

    # Use XGBoost or Cost-Sensitive RF for SHAP (tree-based models work best)
    shap_model_name = st.selectbox("Select model to explain", ['XGBoost', 'Cost-Sensitive RF'], key='shap_model')
    shap_model = models[shap_model_name]

    X_test_df_shap = pd.DataFrame(X_test, columns=X.columns)
    y_test_arr_shap = y_test.values

    shap_fraud_idx = np.where(y_test_arr_shap == 1)[0]
    shap_legit_idx = np.where(y_test_arr_shap == 0)[0]

    shap_choice = st.radio("Pick a transaction to explain:",
                           ["Random Fraud", "Random Legitimate", "Random Any"], horizontal=True)

    if st.button("🔬 Generate SHAP Explanation", use_container_width=True):
        if shap_choice == "Random Fraud":
            idx = shap_fraud_idx[np.random.randint(len(shap_fraud_idx))]
        elif shap_choice == "Random Legitimate":
            idx = shap_legit_idx[np.random.randint(len(shap_legit_idx))]
        else:
            idx = np.random.randint(len(X_test_df_shap))

        sample_row = X_test_df_shap.iloc[[idx]]
        actual = y_test_arr_shap[idx]
        prob_shap = shap_model.predict_proba(sample_row)[0][1]

        st.markdown(f"**Actual Label:** {'🚨 Fraud' if actual == 1 else '✅ Legitimate'} | "
                    f"**Model Probability:** {prob_shap:.1%}")

        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(shap_model)
            shap_values = explainer(sample_row)

        st.markdown("#### Waterfall Chart")
        st.markdown("Each bar shows how much a feature pushed the prediction toward Fraud (red/right) or Legitimate (blue/left).")

        fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
        fig_shap.patch.set_facecolor('#0e1117')
        ax_shap.set_facecolor('#0e1117')

        # Get SHAP values for class 1 (fraud)
        if isinstance(shap_values.values, np.ndarray) and shap_values.values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        shap_vals = sv.values[0]
        feature_vals = sample_row.values[0]
        feat_names = X.columns.tolist()

        # Sort by absolute SHAP value, show top 15
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:15]

        colors_shap = ['#ff4b4b' if v > 0 else '#3a7bd5' for v in shap_vals[top_idx]]
        y_pos = range(len(top_idx))

        ax_shap.barh(y_pos, shap_vals[top_idx][::-1], color=colors_shap[::-1], edgecolor='none', height=0.7)
        ax_shap.set_yticks(y_pos)
        labels = [f"{feat_names[i]} = {feature_vals[i]:.3f}" for i in top_idx][::-1]
        ax_shap.set_yticklabels(labels, color='white', fontsize=9)
        ax_shap.set_xlabel('SHAP Value (impact on fraud prediction)', color='white', fontsize=11)
        ax_shap.set_title(f'Top 15 Feature Contributions ({shap_model_name})', color='white', fontsize=14, fontweight='bold')
        ax_shap.axvline(x=0, color='white', linewidth=0.5, alpha=0.3)
        ax_shap.tick_params(colors='white')
        for spine in ax_shap.spines.values():
            spine.set_color('#333')
        plt.tight_layout()
        st.pyplot(fig_shap)

        st.markdown("---")
        st.markdown("**How to read this:** Red bars pointing right = pushes toward **FRAUD**. Blue bars pointing left = pushes toward **LEGITIMATE**. "
                    "The further the bar extends, the stronger that feature's influence on this specific prediction.")

# ─── Footer ───
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #555; font-size: 0.8rem; padding: 1rem;">
    Credit Card Fraud Detection Dashboard &bull; Predictive Analytics Case Study &bull; Group 20 &bull; VIT Winter 2025-2026
</div>
""", unsafe_allow_html=True)
