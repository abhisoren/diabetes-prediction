# Diabetes Prediction: Binary Classification Project

Predicts whether a patient has diabetes from 8 diagnostic measurements (Pima Indians Diabetes dataset, 768 rows).

## Contents
- `Diabetes_Prediction_Project.ipynb`: full end-to-end notebook

## Pipeline
1. EDA: class balance, distributions, correlations
2. Treat impossible zeros (glucose, blood pressure, skin thickness, insulin, BMI) as missing
3. Stratified train/test split; median imputation + scaling inside a scikit-learn `Pipeline` (no data leakage)
4. 5-fold CV comparison: Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting
5. Grid-search tuning, evaluation on held-out test set (accuracy, recall, ROC-AUC)
6. Decision-threshold tuning for higher recall (screening use case)
7. Permutation feature importance, model saving (`joblib`), `predict_patient()` helper

## Run it
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
jupyter notebook Diabetes_Prediction_Project.ipynb
```
The data loads from `https://github.com/YBIFoundation/Dataset/raw/main/Diabetes.csv`. Offline? Download the CSV and change the `URL` variable.

## Baseline (original notebook)
Logistic regression on raw data: accuracy 0.77, recall for diabetics 0.52.

*Educational project, not a medical diagnostic tool.*
