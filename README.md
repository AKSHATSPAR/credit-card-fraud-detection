# Credit Card Fraud Detection with Imbalanced Data
**Predictive Analytics | Group 20 | Winter Semester 2025-2026**

**Group Members:**
- Akshat Sparsh (23BDS0149)
- Ankit Kumar (23BCE0659)

---

## 🚀 End-to-End Fresh Setup Guide
Welcome! This repository contains the complete execution code, Jupyter Notebook presentations, and automated PDF report generation scripts necessary to recreate our predictive analytics case study from an absolute "fresh laptop" state. 

### Prerequisites
- You must have **Python 3.10** or higher installed. (Install from [python.org](https://www.python.org/downloads/)).
- You must have the dataset. Ensure `creditcard.csv` is located in the same root folder as this README.

### 💨 1-Click Setup Methods

Depending on your Operating System, we have provided automated setup scripts that will create a strict virtual environment, install the exact dependencies, and immediately launch our Interactive Jupyter Notebook Presentation.

**For Windows Users:**
Double-click the `setup_and_run_windows.bat` file.
*Alternatively, open your command prompt (cmd) and type:*
```cmd
setup_and_run_windows.bat
```

**For macOS / Linux Users:**
Open your Terminal, navigate to this directory, and type:
```bash
chmod +x setup_and_run_mac.sh
./setup_and_run_mac.sh
```

---

### 📘 What is inside the Jupyter Notebook?
The notebook launched by our setup scripts (`notebooks/Credit_Card_Fraud_Detection_Case_Study.ipynb`) contains our **entire presentation slide deck** mixed with the actual Python execution cells that generate the results.

You can click `Run All Cells` inside the notebook to watch it:
1. Load and process the highly imbalanced `creditcard.csv` dataset.
2. Apply SMOTE and Cost-Sensitive learning paradigms.
3. Compare Logistic Regression and Random Forests across Precision, Recall, and ROC-AUC.
4. Output the exact Confusion Matrices, ROC Curve Graphs, and Feature Importances shown in our PDF submissions!

### 📄 PDF Report Generation
If you want to view or regenerate the strictly formatted A4 Technical Report PDF, you can run the primary python script from the terminal (Ensure you are in the virtual environment):
```bash
python credit_card_fraud_detection.py
```
This script will produce `23BDS0149_CreditCardFraudDetection.pdf` directly into this folder.

---

### 🖥️ Interactive Dashboard (Streamlit GUI)
We also built a **real-time Fraud Detection Dashboard** using Streamlit. This premium dark-themed web app allows you to:
- Switch between all 4 trained models live
- Drag an interactive **Decision Threshold slider** and watch the Confusion Matrix update in real time
- View ROC Curves, Feature Importances, and Confusion Matrices
- **Predict individual transactions** as Fraud or Legitimate using a Live Predictor form

To launch the dashboard:
```bash
source venv/bin/activate
pip install streamlit -q
streamlit run app.py
```
The dashboard will open at `http://localhost:8501` in your browser automatically.
