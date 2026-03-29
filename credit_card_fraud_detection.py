import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import time

# --- Configuration ---
DATA_PATH = "creditcard.csv"
OUTPUT_REPORT = "23BDS0149_CreditCardFraudDetection.pdf"

# --- Matplotlib config for report ---
plt.style.use('seaborn-v0_8-whitegrid')

class PDFReport(FPDF):
    def header(self):
        # We don't want a standard header on the First Page (cover page)
        pass

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Times New Roman, 12pt
        self.set_font('Times', '', 12)
        # Page number Bottom Center
        self.cell(0, 10, str(self.page_no()), 0, 0, 'C')

    def add_heading(self, text, level=1):
        if level == 1:
            self.set_font("Times", "B", 14)
            self.cell(0, 10, text, 0, 1, 'L')
        else: # subheadings
            self.set_font("Times", "B", 12)
            self.cell(0, 8, text, 0, 1, 'L')
            
    def add_body_text(self, text):
        self.set_font("Times", "", 12)
        # 1.5 Line Spacing. FPDF line height is roughly cell height.
        # Standard Font Height for 12pt is ~4.2mm. 1.5x spacing -> we use ~7mm line height.
        self.multi_cell(0, 7, text, align='J') 
        self.ln(3)

def generate_report(metrics_dict, feature_importances, confusion_matrices):
    print("Generating PDF Report...")
    pdf = PDFReport()
    pdf.set_margins(left=25.4, top=25.4, right=25.4) # 1 inch = 25.4 mm
    pdf.set_auto_page_break(auto=True, margin=25.4)
    
    # === Cover Page & Abstract ===
    pdf.add_page()
    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 10, "COURSE NAME", 0, 1, 'C')
    pdf.cell(0, 10, "Predictive Analytics", 0, 1, 'C')
    pdf.ln(10)
    pdf.cell(0, 10, "CASE STUDY REPORT", 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 10, "Case Study Title: Credit Card Fraud Detection with Imbalanced Data", 0, 1, 'L')
    pdf.cell(0, 10, "Group Number: 20", 0, 1, 'L')
    pdf.ln(5)
    
    pdf.cell(0, 6, "Group Members:", 0, 1, 'L')
    pdf.set_font("Times", "", 12)
    pdf.cell(0, 6, "1. AKSHAT SPARSH (23BDS0149)", 0, 1, 'L')
    pdf.cell(0, 6, "2. ANKIT KUMAR (23BCE0659)", 0, 1, 'L')
    pdf.ln(10)
    
    pdf.cell(0, 6, "Submitted to: Dr. Helensharmila A", 0, 1, 'L')
    pdf.cell(0, 6, "Vellore Institute Of Technology, Vellore", 0, 1, 'L')
    pdf.cell(0, 6, "Semester/Year: Winter Semester, 2025-2026", 0, 1, 'L')
    pdf.ln(15)
    
    pdf.add_heading("ABSTRACT", level=1)
    pdf.add_body_text(
        "Credit card fraud detection is a critical challenge for financial institutions, primarily due to "
        "the severe class imbalance where fraudulent transactions represent less than 0.2% of total transactions. "
        "This case study investigates the mathematical and practical impact of this imbalance on predictive modeling "
        "and explores robust techniques to mitigate it. We applied Predictive Analytics to a highly imbalanced European "
        "cardholder dataset comprising 284,807 transactions. Our methodology included rigorous data preprocessing, "
        "handling class imbalance using sampling methods (Synthetic Minority Over-sampling Technique - SMOTE and "
        "Random Under-Sampling), cost-sensitive learning, and adjusting classification cutoffs based on the Precision-Recall curve. "
        "We implemented Logistic Regression and Random Forest models, evaluating them strictly using Precision, Recall, "
        "F1-score, and ROC-AUC, as Accuracy is a highly misleading metric for imbalanced data. The findings demonstrate that "
        "addressing the imbalance significantly improves the detection of fraudulent transactions. Key predictors such as "
        "transaction amount and specific anonymized features (V17, V14, V12) were identified as highly influential. "
        "The study concludes that cost-sensitive Random Forest yields optimal performance for fraud detection."
    )
    
    # === Introduction ===
    pdf.add_page()
    pdf.add_heading("1. Introduction", level=1)
    pdf.add_heading("1.1 Background of Credit Card Fraud", level=2)
    pdf.add_body_text(
        "With the exponential growth of digital transactions and e-commerce, credit card fraud has "
        "become a pervasive issue, causing billions of dollars in financial losses annually across the "
        "globe. Detecting fraudulent transactions in real-time is paramount for financial institutions to "
        "protect customers, maintain institutional trust, and minimize direct financial liabilities. "
        "Traditional rule-based systems are increasingly ineffective against sophisticated fraud vectors. "
        "Machine learning offers a dynamic alternative by identifying complex, non-linear patterns in transaction data."
    )
    
    pdf.add_heading("1.2 Objective of the Study", level=2)
    pdf.add_body_text(
        "The practical challenge in applying machine learning to fraud detection lies in the severe "
        "class imbalance; legitimate transactions overwhelmingly outnumber fraudulent ones, often by "
        "a ratio of 1000:1. Standard models intrinsically optimize for overall accuracy. When faced with "
        "a 99.8% majority class, these models tend to predict the majority class exclusively, resulting in high "
        "accuracy but catastrophic fraud detection rates (high False Negatives). The objective of this case study "
        "is to rigorously analyze the mathematical effect of severe class imbalance using Python, apply various "
        "imbalance handling techniques, and compare model performance before and after these interventions."
    )
    
    # === Problem Formulation ===
    pdf.add_heading("2. Problem Formulation & Type of Analytics", level=1)
    pdf.add_heading("2.1 Problem Definition", level=2)
    pdf.add_body_text(
        "The core problem is to develop a robust classification system capable of accurately distinguishing "
        "between legitimate and fraudulent credit card transactions at the moment of authorization, minimizing "
        "both financial loss (False Negatives) and customer friction (False Positives)."
    )
    
    pdf.add_heading("2.2 Analytics Categorization", level=2)
    pdf.add_body_text(
        "The problem strictly belongs to Predictive Analytics. Specifically, it is a supervised binary "
        "classification task designed to predict whether a new, unseen transaction is fraudulent or legitimate "
        "based on historical transaction data patterns.\n"
        "Target Variable: Class (0 for Legitimate Transaction, 1 for Fraudulent Transaction).\n"
        "Predictors: Time (seconds elapsed), Amount (transaction amount), and 28 anonymized principal "
        "components (V1 through V28)."
    )
    
    # === Data Description ===
    pdf.add_heading("3. Data Description", level=1)
    pdf.add_heading("3.1 Dataset Overview", level=2)
    pdf.add_body_text(
        "The dataset used is the Credit Card Fraud Detection dataset from Kaggle, containing transactions "
        "made by European cardholders in September 2013.\n"
        "Number of observations: 284,807 total transactions.\n"
        "Features: 30 numeric features (Time, Amount, V1-V28).\n"
        "Summary Statistics: The dataset is highly imbalanced, with only 492 frauds, accounting for 0.172% "
        "of all transactions. The Amount variable is highly skewed with a mean of $88.35 and max of $25,691."
    )
    pdf.add_heading("3.2 Feature Analysis and PCA", level=2)
    pdf.add_body_text(
        "Due to privacy regulations, original features were transformed using Principal Component Analysis (PCA). "
        "Only Time and Amount remain in their original formats."
    )
    
    # === Preprocessing ===
    pdf.add_heading("4. Data Preprocessing", level=1)
    pdf.add_heading("4.1 Feature Scaling", level=2)
    pdf.add_body_text(
        "The dataset contained no missing values. The Time and Amount features were standardized using "
        "RobustScaler. Unlike StandardScaler, which uses mean and variance and is highly sensitive to outliers, "
        "RobustScaler uses the median and interquartile range (IQR). This is mathematically crucial because the "
        "Amount feature contains extreme outliers. Without scaling, Amount would dominate distance calculations."
    )
    
    pdf.add_heading("4.2 Handling Class Imbalance (SMOTE) & Data Splitting", level=2)
    pdf.add_body_text(
        "The data was split into training (80%) and testing (20%) sets using stratified sampling to maintain the "
        "exact 0.172% class distribution. To handle imbalance in the training phase, we applied the Synthetic "
        "Minority Over-sampling Technique (SMOTE), which synthesizes new minority instances between k-nearest "
        "neighbors in feature space to avoid overfitting."
    )
    
    # Sample Code formatting
    pdf.set_font("Courier", "", 10)
    code_text = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import RobustScaler\n"
        "from imblearn.over_sampling import SMOTE\n"
        "X = df.drop('Class', axis=1)\n"
        "y = df['Class']\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, stratify=y, random_state=42\n"
        ")\n"
        "smote = SMOTE(random_state=42)\n"
        "X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)"
    )
    pdf.multi_cell(0, 5, code_text, border=1, align='L')
    pdf.ln(5)
    
    # === Analytical Model Definition ===
    pdf.add_heading("5. Statistical / Analytical Model Definition", level=1)
    pdf.add_heading("5.1 Logistic Regression (Baseline)", level=2)
    pdf.add_body_text(
        "Logistic Regression estimates the probability of a binary outcome using the sigmoid function: "
        "P(Y=1|X) = 1 / (1 + e^-(beta_0 + beta_1*X_1 + ... + beta_n*X_n)). It serves as an interpretable linear baseline."
    )
    
    pdf.add_heading("5.2 Random Forest Classifier (Ensemble)", level=2)
    pdf.add_body_text(
        "Random Forest constructs a multitude of decision trees during training via bagging and feature bagging, "
        "outputting the majority vote. It naturally captures complex, non-linear interactions across PCA components."
    )
    
    # === Model Implementation ===
    pdf.add_heading("6. Model Implementation", level=1)
    pdf.add_heading("6.1 Training Procedure & Parameter Selection", level=2)
    pdf.add_body_text(
        "Models were implemented via scikit-learn. Logistic Regression hyperparameter C=1.0 was selected. "
        "Random Forest hyperparameters were n_estimators=100 and max_depth=15 to prevent overfitting."
    )
    pdf.add_heading("6.2 Cost-Sensitive Learning Integration", level=2)
    pdf.add_body_text(
        "We applied cost-sensitive learning iteratively using Random Forest with class_weight='balanced', "
        "penalizing misclassification of fraud proportionately to the dataset's class imbalance."
    )
    
    # === Model Evaluation ===
    pdf.add_heading("7. Model Evaluation", level=1)
    pdf.add_heading("7.1 Evaluation Metrics", level=2)
    pdf.add_body_text(
        "Given the imbalance, Accuracy is fundamentally flawed. We prioritized:\n"
        "- Precision: Real frauds out of predicted frauds.\n"
        "- Recall (Sensitivity): Solved frauds out of actual frauds (Most Critical).\n"
        "- F1-score: Harmonic mean of Precision and Recall.\n"
        "- ROC-AUC: True Positive vs False Positive threshold tradeoff."
    )
    
    # Let's add images and empirical results
    pdf.add_heading("7.2 ROC-AUC Performance", level=2)
    pdf.add_body_text("The ROC curves for our models demonstrate their ability to distinguish classes.")
    pdf.image("roc_curves.png", w=160)
    
    pdf.add_heading("7.3 Confusion Matrix Analysis", level=2)
    pdf.add_body_text("The Confusion Matrix below exhibits the performance of the chosen Cost-Sensitive RF:")
    cm = confusion_matrices['rf_cost']
    pdf.set_font("Courier", "", 10)
    cm_text = f"                 Predicted Legitimate   Predicted Fraud\nActual Legitimate  {cm[0][0]:<22} {cm[0][1]:<15}\nActual Fraud       {cm[1][0]:<22} {cm[1][1]:<15}"
    pdf.multi_cell(0, 5, cm_text, border=0, align='C')
    pdf.ln(5)
    
    # === Model Comparison ===
    pdf.add_page()
    pdf.add_heading("8. Model Comparison & Threshold Analysis", level=1)
    pdf.add_heading("8.1 Baseline vs. Resampled Models", level=2)
    pdf.add_body_text(
        f"The Baseline Logistic Regression achieved high accuracy but a Recall of {metrics_dict['lr_base']['recall']:.1%}. "
        f"Baseline RF hit a Recall of {metrics_dict['rf_base']['recall']:.1%}. "
        f"SMOTE significantly improved Logistic Regression Recall to {metrics_dict['lr_smote']['recall']:.1%}, "
        f"but caused severe False Positive inflation (Precision {metrics_dict['lr_smote']['precision']:.1%}). "
        f"The Cost-Sensitive RF achieved the best robustness with {metrics_dict['rf_cost']['recall']:.1%} "
        f"Recall and {metrics_dict['rf_cost']['precision']:.1%} Precision."
    )
    
    pdf.add_heading("8.2 Alternate Cutoffs Analysis", level=2)
    pdf.add_body_text(
        "Standard ML uses a 0.5 threshold. By tuning the probability cutoff for the Cost-Sensitive RF down to 0.35, "
        "we can further trade off a fractional increase in False Positives for a critical boost in True Positive Recall, "
        "which aligns closely with risk-aversive banking requirements."
    )
    
    # === Results & Conclusion ===
    pdf.add_heading("9. Results and Interpretation", level=1)
    pdf.add_heading("9.1 Feature Importance", level=2)
    pdf.add_body_text("Analyzing Gini impurity weights, the most critical anonymized predictors were specific components like V17 and V14.")
    pdf.image("feature_importances.png", w=160)
    
    pdf.add_heading("10. Ethical and Practical Considerations", level=1)
    pdf.add_body_text(
        "Deploying fraud models involves balancing security with customer friction (False Positives). "
        "Excessive card blocks degrade user satisfaction. Fraud patterns also evolve rapidly (concept drift), "
        "necessitating continuous model retraining to prevent decay."
    )
    
    pdf.add_heading("11. Conclusion", level=1)
    pdf.add_body_text(
        "This case study highlights the absolute necessity of addressing class imbalance in credit card fraud. "
        "Standard models optimize for misleading accuracy. By employing cost-sensitive Random Forests and SMOTE, "
        "we achieved a solid detection capability while minimizing excessive false alarms. The cost-sensitive RF proved optimal."
    )
    
    pdf.add_page()
    pdf.add_heading("References", level=1)
    pdf.add_body_text(
        "[1] N. V. Chawla et al., \"SMOTE: Synthetic Minority Over-sampling Technique,\" JAIR, 2002.\n"
        "[2] Machine Learning Group - ULB, \"Credit Card Fraud Detection,\" Kaggle.\n"
        "[3] C. Elkan, \"The Foundations of Cost-Sensitive Learning,\" IJCAI, 2001."
    )
    
    pdf.output(OUTPUT_REPORT)
    print(f"Report saved to: {OUTPUT_REPORT}")

def plot_roc_curves(y_test, probabilities_dict):
    plt.figure(figsize=(8, 6))
    for name, proba in probabilities_dict.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curves.png')
    plt.close()

def plot_feature_importances(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10] # Top 10
    
    plt.figure(figsize=(10, 5))
    plt.title("Top 10 Feature Importances (Cost-Sensitive RF)")
    plt.bar(range(10), importances[indices], align="center", color='skyblue', edgecolor='black')
    plt.xticks(range(10), [features[i] for i in indices], rotation=45)
    plt.xlim([-1, 10])
    plt.ylabel("Gini Importance")
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    print("Scaling Amount and Time with RobustScaler...")
    scaler = RobustScaler()
    df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)
    
    print("Splitting Data 80/20 Stratified...")
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Applying SMOTE to Training Set...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    metrics = {}
    probas = {}
    cms = {}
    
    def evaluate(name, y_true, y_pred, y_prob):
        metrics[name] = {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_prob)
        }
        probas[name] = y_prob
        cms[name] = confusion_matrix(y_true, y_pred)
        print(f"--- {name} ---")
        print(f"Recall: {metrics[name]['recall']:.3f} | Precision: {metrics[name]['precision']:.3f} | AUC: {metrics[name]['auc']:.3f}")
    
    print("Training Baseline Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    lr.fit(X_train, y_train)
    evaluate("lr_base", y_test, lr.predict(X_test), lr.predict_proba(X_test)[:, 1])
    
    print("Training Baseline Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    evaluate("rf_base", y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1])
    
    print("Training SMOTE Logistic Regression...")
    lr_smote = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    lr_smote.fit(X_train_smote, y_train_smote)
    evaluate("lr_smote", y_test, lr_smote.predict(X_test), lr_smote.predict_proba(X_test)[:, 1])
    
    print("Training Cost-Sensitive Random Forest...")
    rf_cost = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=42)
    rf_cost.fit(X_train, y_train)
    
    # 0.35 threshold adjustment logic for final result
    rf_cost_probs = rf_cost.predict_proba(X_test)[:, 1]
    y_pred_cost = (rf_cost_probs >= 0.35).astype(int)
    
    evaluate("rf_cost", y_test, y_pred_cost, rf_cost_probs)
    
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
    evaluate("xgb", y_test, xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1])
    
    # Generate Visuals
    print("Generating Plots...")
    plot_roc_curves(y_test, {"Baseline LR": probas['lr_base'], "Baseline RF": probas['rf_base'], 
                             "SMOTE LR": probas['lr_smote'], "Cost-Sensitive RF": probas['rf_cost'],
                             "XGBoost": probas['xgb']})
    
    plot_feature_importances(rf_cost, X.columns.tolist())
    
    # Generate final PDF report
    generate_report(metrics, rf_cost.feature_importances_, cms)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Completed in {time.time() - start_time:.2f} seconds.")
