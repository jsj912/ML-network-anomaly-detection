import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading combined dataset...")
combined = pd.read_csv("combined_cic_ctu.csv", low_memory=False)

# Inspect label values
print("\nUnique values in 'label' column before cleaning:")
print(combined['label'].unique()[:20])

# Normalize label column
def normalize_label(x):
    x_str = str(x).upper().strip()
    if "BENIGN" in x_str or "NORMAL" in x_str or x_str == "0":
        return 0
    elif "MALICIOUS" in x_str or "ATTACK" in x_str or x_str == "1":
        return 1
    else:
        return np.nan

combined['label'] = combined['label'].apply(normalize_label)

# Drop missing labels
combined = combined.dropna(subset=['label'])
combined['label'] = combined['label'].astype(int)

print("\nCleaned label distribution:")
print(combined['label'].value_counts())

# Convert protocol column to numeric
if 'protocol' in combined.columns:
    combined['protocol'] = pd.to_numeric(combined['protocol'], errors='coerce').fillna(0)
    print("\nConverted 'protocol' column to numeric.")
else:
    print("No 'protocol' column found, skipping numeric conversion.")

# Balance dataset 
print("Balancing dataset...")
benign = combined[combined['label'] == 0].sample(n=300000, random_state=42, replace=True)
mal = combined[combined['label'] == 1].sample(n=300000, random_state=42, replace=True)
balanced = pd.concat([benign, mal])
print("Balanced dataset shape:", balanced.shape)


# Split into features/labels

X = balanced.drop(columns=['label'])
y = balanced['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print("Train:", X_train.shape, "Test:", X_test.shape)


print("\nTraining LightGBM model...")

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------
# Evaluate model
# -------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, target_names=['Benign', 'Malicious'])
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\nMODEL TRAINED SUCCESSFULLY")
print(report)
print(f"ROC-AUC: {roc_auc:.3f}")


# Save model + metrics

joblib.dump(model, 'baseline_lightgbm.pkl')
with open('metrics_lightgbm.txt', 'w') as f:
    f.write(report)
    f.write(f"\nROC-AUC: {roc_auc:.3f}\n")

print("Model saved as 'baseline_lightgbm.pkl'")


# Visualize Confusion Matrix

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('LightGBM Confusion Matrix')
plt.tight_layout()
plt.savefig("lightgbm_confusion_matrix.png", dpi=300)
plt.show()


# Feature Importance

importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='coolwarm')
plt.title('Top 15 Feature Importances (LightGBM)')
plt.tight_layout()
plt.savefig("lightgbm_feature_importance.png", dpi=300)
plt.show()

print("Saved visualizations: confusion matrix + feature importance")
