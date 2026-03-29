@echo off
echo Starting Deep Setup for Credit Card Fraud Detection Project...

:: 1. Ensure Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python could not be found. Please install Python 3.10+ from python.org and ensure it is added to your PATH.
    pause
    exit /b 1
)
echo [OK] Python is installed.

:: 2. Setup Virtual Environment
echo Creating isolated virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

:: 3. Install Dependencies
echo Installing exact dependencies from requirements.txt (this may take a minute)...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 4. Run Notebook Environment
echo Setup Complete!
echo Launching the interactive Data Science Jupyter Notebook...
jupyter notebook notebooks\Credit_Card_Fraud_Detection_Case_Study.ipynb
pause
