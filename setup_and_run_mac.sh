#!/bin/bash

echo "Starting Deep Setup for Credit Card Fraud Detection Project..."

# 1. Ensure Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found. Please install Python 3.10+ from python.org"
    exit 1
fi

echo "✓ Python3 is installed."

# 2. Setup Virtual Environment
echo "Creating isolated virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Install Dependencies
echo "Installing exact dependencies from requirements.txt (this may take a minute)..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run Notebook Environment
echo "Setup Complete!"
echo "Launching the interactive Data Science Jupyter Notebook..."
jupyter notebook notebooks/Credit_Card_Fraud_Detection_Case_Study.ipynb
