# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from datetime import datetime, timedelta
from tqdm import tqdm
from math import radians, sin, cos, sqrt, atan2

# Set random seed for reproducibility
np.random.seed(42)

#1. Data Understanding

# Load train and test datasets 
train_data = pd.read_csv("C:\Users\Downloads\archive\fraudTrain.csv")  
test_data = pd.read_csv("C:\Users\Downloads\archive\fraudTest.csv")    

print("Training Dataset Shape:", train_data.shape)
print("Test Dataset Shape:", test_data.shape)
print("\nTraining Columns:", train_data.columns)

# Visualize class imbalance in training data (before SMOTE)
plt.figure(figsize=(6, 4))
sns.countplot(x='is_fraud', data=train_data)
plt.title('Class Distribution in Training Data (Before SMOTE)')
plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
plt.ylabel('Count')
plt.show()

# Displaying class distribution
print("\nClass Distribution in Training Data:\n", train_data['is_fraud'].value_counts(normalize=True))
print("\nClass Distribution in Test Data:\n", test_data['is_fraud'].value_counts(normalize=True))

# 2. Data Preprocessing

# Handle missing values
train_data = train_data.dropna()
test_data = test_data.dropna()

# Convert timestamp to datetime and extract features
train_data['trans_date_trans_time'] = pd.to_datetime(train_data['trans_date_trans_time'])
test_data['trans_date_trans_time'] = pd.to_datetime(test_data['trans_date_trans_time'])

# Extract time-based features
train_data['hour'] = train_data['trans_date_trans_time'].dt.hour
train_data['day_of_week'] = train_data['trans_date_trans_time'].dt.dayofweek
train_data['month'] = train_data['trans_date_trans_time'].dt.month

test_data['hour'] = test_data['trans_date_trans_time'].dt.hour
test_data['day_of_week'] = test_data['trans_date_trans_time'].dt.dayofweek
test_data['month'] = test_data['trans_date_trans_time'].dt.month

# Frequency encoding for categorical variables
train_data['merchant_freq'] = train_data.groupby('merchant')['merchant'].transform('count')
train_data['user_id_freq'] = train_data.groupby('cc_num')['cc_num'].transform('count')
train_data['category_freq'] = train_data.groupby('category')['category'].transform('count')

test_data['merchant_freq'] = test_data.groupby('merchant')['merchant'].transform('count')
test_data['user_id_freq'] = test_data.groupby('cc_num')['cc_num'].transform('count')
test_data['category_freq'] = test_data.groupby('category')['category'].transform('count')

# Additional features for better understanding

# A. Transaction velocity (transactions per user in last hour)
train_data = train_data.sort_values(['cc_num', 'trans_date_trans_time'])
test_data = test_data.sort_values(['cc_num', 'trans_date_trans_time'])

train_data['time_diff'] = train_data.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds().fillna(3600) / 3600
train_data['velocity'] = train_data.groupby('cc_num').cumcount() / (train_data.groupby('cc_num')['time_diff'].cumsum() + 1e-6)

test_data['time_diff'] = test_data.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds().fillna(3600) / 3600
test_data['velocity'] = test_data.groupby('cc_num').cumcount() / (test_data.groupby('cc_num')['time_diff'].cumsum() + 1e-6)

# B. Standardized amount per user (z-score of amount)
train_data['amt_z_score'] = train_data.groupby('cc_num')['amt'].transform(lambda x: (x - x.mean()) / x.std())
test_data['amt_z_score'] = test_data.groupby('cc_num')['amt'].transform(lambda x: (x - x.mean()) / x.std())

# C. Flag for unusual hours (e.g., 1-5 AM might be suspicious)
train_data['is_unusual_hour'] = train_data['hour'].apply(lambda x: 1 if 1 <= x <= 5 else 0)
test_data['is_unusual_hour'] = test_data['hour'].apply(lambda x: 1 if 1 <= x <= 5 else 0)

# D. Ratio of amount to user's average amount
train_data['amt_to_user_avg'] = train_data['amt'] / train_data.groupby('cc_num')['amt'].transform('mean')
test_data['amt_to_user_avg'] = test_data['amt'] / test_data.groupby('cc_num')['amt'].transform('mean')

# E. Transaction count in last 24 hours per user
def get_recent_count(df):
    recent_counts = []
    for _, group in df.groupby('cc_num'):
        group = group.sort_values('trans_date_trans_time')
        counts = []
        for idx, row in group.iterrows():
            time_threshold = row['trans_date_trans_time'] - timedelta(hours=24)
            count = group[(group['trans_date_trans_time'] >= time_threshold) & (group['trans_date_trans_time'] <= row['trans_date_trans_time'])].shape[0]
            counts.append(count)
        recent_counts.extend(counts)
    return recent_counts

train_data['transaction_count_recent'] = get_recent_count(train_data)
test_data['transaction_count_recent'] = get_recent_count(test_data)

# 7. Standard deviation of amounts per user
train_data['amt_std_dev'] = train_data.groupby('cc_num')['amt'].transform('std')
test_data['amt_std_dev'] = test_data.groupby('cc_num')['amt'].transform('std')

# Verify features
print("\nVelocity Feature Summary (Train):\n", train_data['velocity'].describe())
print("\nTransaction Count Recent Summary (Train):\n", train_data['transaction_count_recent'].describe())

# Check for NaN or infinite values
train_data = train_data.fillna(0).replace([np.inf, -np.inf], 0)
test_data = test_data.fillna(0).replace([np.inf, -np.inf], 0)

# Select features (excluding 'distance' due to low importance)
features = ['amt', 'hour', 'day_of_week', 'month', 'merchant_freq', 'user_id_freq', 'category_freq', 'velocity', 'amt_z_score', 'is_unusual_hour', 'amt_to_user_avg', 'transaction_count_recent', 'amt_std_dev']
X_train = train_data[features]
y_train = train_data['is_fraud']
X_test = test_data[features]
y_test = test_data['is_fraud']

# Check for data integrity
print("\nAny NaN in X_train:\n", X_train.isna().sum())
print("Any NaN in X_test:\n", X_test.isna().sum())

# Feature Correlation Matrix (before scaling)
plt.figure(figsize=(14, 10))
corr_matrix = X_train.corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Feature Correlation Matrix (Training Data)')
plt.show()

# Display correlation matrix
print("\nFeature Correlation Matrix:\n", corr_matrix)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3. Handle Class Imbalance (Training Data Only)

smote = SMOTE(random_state=42, sampling_strategy=1.0)  # 1:1 ratio
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Check class distribution after SMOTE
print("\nAfter SMOTE, Class Distribution in Training Data:\n", pd.Series(y_train_resampled).value_counts())

# Visualize class distribution after SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_resampled)
plt.title('Class Distribution in Training Data (After SMOTE)')
plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
plt.ylabel('Count')
plt.show()


# 4. Model Building (XGBoost with GridSearchCV)

# Define parameter grid for GridSearchCV
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.03, 0.05],
    'n_estimators': [200, 300, 400],
    'scale_pos_weight': [len(y_train[y_train == 0]) / len(y_train[y_train == 1]) * 1.0,
                        len(y_train[y_train == 0]) / len(y_train[y_train == 1]) * 1.5,
                        len(y_train[y_train == 0]) / len(y_train[y_train == 1]) * 2.0]
}

# Initialize base model
base_model = XGBClassifier(random_state=42, verbosity=1, n_jobs=-1)

# Perform GridSearchCV
print("Performing GridSearchCV...")
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

# Fit GridSearchCV
grid_search.fit(X_train_resampled, y_train_resampled)

# Print best parameters and score
print("\nBest Parameters:", grid_search.best_params_)
print("Best F1-Macro Score:", grid_search.best_score_)

# Use best model
model = grid_search.best_estimator_

# Train with tqdm progress bar
print("Training Best Model...")
with tqdm(total=100, desc="Training Progress", unit="%", leave=True) as pbar:
    model.fit(X_train_resampled, y_train_resampled, verbose=True)
    pbar.update(100)

# Cross-validation to check generalization
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='f1_macro')
print("\nCross-Validation F1-Macro Scores:", cv_scores)
print("Mean CV F1-Macro Score:", cv_scores.mean())

# Predictions on test set with threshold optimization
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"\nOptimal Threshold for F1: {optimal_threshold:.2f}")
y_pred = (y_pred_proba >= optimal_threshold).astype(int)


# 5. Model Evaluation

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print("\nClassification Report (Test Set):\n", classification_report(y_test, y_pred))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("\nROC-AUC Score (Test Set):", roc_auc)

# Check prediction distribution
print("\nPrediction Distribution:\n", pd.Series(y_pred).value_counts(normalize=True))

# Feature Importance
import xgboost
xgboost.plot_importance(model)
plt.title('Feature Importance')
plt.show()

# 6. Model Explainability (SHAP)

# Sample a subset of test data for SHAP (to reduce computation time)
sample_size = 1000  # Adjust based on your computational resources
X_test_sample_indices = np.random.choice(X_test_scaled.shape[0], sample_size, replace=False)
X_test_sample = X_test_scaled[X_test_sample_indices]
y_test_sample = y_test.iloc[X_test_sample_indices]
y_pred_sample = y_pred[X_test_sample_indices]

# Compute SHAP values on the sample
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sample)

# Summary plot
shap.summary_plot(shap_values, X_test_sample, feature_names=features)

# Explain misclassifications (False Positives)
false_positives = (y_test_sample == 0) & (y_pred_sample == 1)
fp_indices = np.where(false_positives)[0]

