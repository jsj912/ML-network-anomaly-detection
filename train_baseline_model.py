import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Load features
df = pd.read_csv('features.csv')

# Simulate labels (replace later with actual dataset labels)
np.random.seed(42)
df['label'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])  # 0=benign, 1=malicious

# Separate features and target
X = df.drop(columns=['label'])
y = df['label']

# Ensure only numeric columns
X = X.select_dtypes(include=[np.number])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train Gradient Boosting model
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
report = classification_report(y_test, y_pred, target_names=['Benign', 'Malicious'])
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("Model trained successfully.")
print(report)
print(f"ROC-AUC: {roc_auc:.3f}")
print("Confusion Matrix:")
print(cm)

# Save model and metrics
joblib.dump(model, 'baseline_model.pkl')

with open('metrics.txt', 'w') as f:
    f.write(report)
    f.write(f"\nROC-AUC: {roc_auc:.3f}\n")
    f.write(f"\nConfusion Matrix:\n{cm}\n")

print("Model saved as 'baseline_model.pkl' and metrics saved as 'metrics.txt'")
