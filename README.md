# GrowthLink

Fraud Detection Model
A machine learning model using XGBoost to detect fraudulent transactions, handling class imbalance with SMOTE and SHAP for explainability.

Installation
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap seaborn matplotlib tqdm

Download fraudTrain.csv and fraudTest.csv from Kaggle and update paths in credit_fraud_det.py

Usage
Run credit_fraud_det.py in a Python supported environment to process data, train the model, and view results.

Results
Class Distribution: Before SMOTE (1.2M non-fraud, ~6K fraud); After SMOTE (1.2M each).
Confusion Matrix: TN: 553316, FP: 258, FN: 1954, TP: 191.
Classification Report: Class 0 (P: 1.00, R: 1.00, F1: 1.00); Class 1 (P: 0.43, R: 0.09, F1: 0.15); Accuracy: 1.00; Macro F1: 0.57.
ROC-AUC: 0.6615.
Prediction Distribution: 0: 0.999, 1: 0.001.
Feature Importance: amt (6550), category_freq (5648), user_id_freq (5624).

Methodology
Preprocessed data with time-based and engineered features.
Balanced with SMOTE (1:1).
Tuned XGBoost with GridSearchCV.
Evaluated with metrics and SHAP (sampled 1000 instances).

Notes
Accuracy Limitation: Accuracy is 1.00 but misleading due to imbalanced test data (~99.6% non-fraud), with poor fraud recall (0.09).

Improvements
Boost fraud recall (0.09) and F1 (0.15) by expanding param_grid or adding user_fraud_rate.
Optimize SHAP with smaller sample_size if slow.


