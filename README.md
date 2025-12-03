Adult Income Prediction – End-to-End Machine Learning Capstone Project
Author: Jyothipriya Ramavath • Master’s in Computer Information Systems, CBU
Skills: Python • Pandas • Scikit-Learn • FastAPI • Data Engineering • EDA • ML Deployment
 Project Overview

This capstone project predicts whether an individual earns more than $50K/year using the UCI Adult Census Dataset.
It demonstrates a complete end-to-end machine learning pipeline:

Data Cleaning

Exploratory Data Analysis

Feature Engineering

Model Development

ML Pipeline (OneHotEncoder + StandardScaler)

Model Evaluation

Model Exporting (joblib)

API Deployment using FastAPI

Testing using Postman

This project is designed to reflect real-world Data Engineering & ML workflows.

📁 Project Structure
adult-income-project/
│
├── data/
│   └── adult/
│        ├── adult.data
│        └── cleaned_adult.csv
│
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
│
├── models/
│   └── final_model.joblib
│
└── app/
    └── main.py  (FastAPI app)

🧹 1. Data Preparation

Performed in 01_data_prep.ipynb:

Applied official UCI column names

Cleaned missing values (" ?")

Removed incomplete rows

Transformed income to binary (0 or 1)

Saved cleaned file as cleaned_adult.csv

📊 2. Exploratory Data Analysis

Performed in 02_eda.ipynb:

Age and work hour distributions

Income imbalance visualization

Key correlations:

Education ↑ → Income ↑

Married people → higher earnings

Gender income gap visible

🤖 3. Machine Learning Modeling

Performed in 03_modeling.ipynb:

Models Used:

Logistic Regression (Baseline)

Random Forest Classifier (Final Model)

Preprocessing:

Numeric → StandardScaler

Categorical → OneHotEncoder

Combined using ColumnTransformer

Entire workflow wrapped in Pipeline

Evaluation:

Accuracy

F1 Score

Classification Report

Final Model: Random Forest (best F1 score)
Exported as: final_model.joblib

🚀 4. Deployment using FastAPI

FastAPI app (app/main.py) includes:

Model loading with joblib

Pydantic schema for validation

/predict endpoint

Returns prediction + probability

Tested using Postman

Run locally:

uvicorn app.main:app --reload


Test with:

POST http://127.0.0.1:8000/predict

🛠️ Tech Stack

Python

Pandas, NumPy

Matplotlib

Scikit-Learn

FastAPI

Uvicorn

Joblib

Postman

VS Code

🏆 Outcome

A fully deployed machine learning model capable of real-time inference.
This project demonstrates strong skills in:

Data Engineering

Data Cleaning

EDA

Machine Learning

Model Deployment

API Development
