import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

cic_folder = Path("cic ids 2017 kaggle")
cic_files = list(cic_folder.glob("*.csv"))
cic_dfs = []
for file in cic_files:
    print(f"Loading {file.name} ...")
    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    if 'label' in df.columns:
        cic_dfs.append(df)
    else:
        print(f"Skipping {file.name} (no label column)")
cic = pd.concat(cic_dfs, ignore_index=True)
print(f"Combined CIC-IDS2017 shape: {cic.shape}")

ctu_folder = Path("ctu 13 kaggle")
ctu_files = list(ctu_folder.glob("*.parquet"))
ctu_dfs = []
for file in ctu_files:
    print(f"Loading {file.name} ...")
    df = pd.read_parquet(file)
    df.columns = df.columns.str.strip().str.lower()
    if 'label' in df.columns:
        ctu_dfs.append(df)
    else:
        print(f"Skipping {file.name} (no label column)")

ctu = pd.concat(ctu_dfs, ignore_index=True)
print(f"Combined CTU-13 shape: {ctu.shape}")

protocol_candidates = [c for c in cic.columns if 'proto' in c or 'protocol' in c]
if len(protocol_candidates) == 0:
    print("No protocol column found — creating dummy 'protocol' column as 0.")
    cic['protocol'] = 0
else:
    cic['protocol'] = cic[protocol_candidates[0]]

# Select safe subset of columns (only those that exist)
cic_feature_candidates = [
    'flow_duration', 'total_fwd_packets', 'total_backward_packets',
    'total_length_of_fwd_packets', 'total_length_of_bwd_packets',
    'flow_iat_mean', 'flow_iat_std', 'protocol', 'label'
]
cic_subset = cic[[c for c in cic_feature_candidates if c in cic.columns]].copy()

rename_map = {
    'flow_duration': 'duration',
    'total_fwd_packets': 'fwd_pkts',
    'total_backward_packets': 'bwd_pkts',
    'total_length_of_fwd_packets': 'fwd_bytes',
    'total_length_of_bwd_packets': 'bwd_bytes',
}
cic_subset.rename(columns=rename_map, inplace=True)

# Fill missing feature columns with 0 (to align with CTU)
for col in ['duration', 'fwd_pkts', 'bwd_pkts', 'fwd_bytes', 'bwd_bytes', 'flow_iat_mean', 'flow_iat_std']:
    if col not in cic_subset.columns:
        cic_subset[col] = 0

print("CIC columns standardized:", list(cic_subset.columns))


ctu_feature_candidates = ['dur', 'totpkts', 'totbytes', 'srcbytes', 'proto', 'label']
ctu_subset = ctu[[c for c in ctu_feature_candidates if c in ctu.columns]].copy()

# Rename CTU columns to align with CIC naming
ctu_subset.rename(columns={
    'dur': 'duration',
    'totpkts': 'fwd_pkts',
    'totbytes': 'fwd_bytes',
    'srcbytes': 'bwd_bytes',
    'proto': 'protocol'
}, inplace=True)

# Add missing IAT columns
for col in ['flow_iat_mean', 'flow_iat_std']:
    ctu_subset[col] = 0

# Ensure column order matches CIC
final_columns = ['duration', 'fwd_pkts', 'bwd_pkts', 'fwd_bytes', 'bwd_bytes',
                 'flow_iat_mean', 'flow_iat_std', 'protocol', 'label']
ctu_subset = ctu_subset.reindex(columns=final_columns, fill_value=0)

print("CTU columns standardized:", list(ctu_subset.columns))

combined = pd.concat([cic_subset, ctu_subset], ignore_index=True)
print(f"Final combined dataset shape: {combined.shape}")
combined_path = "combined_cic_ctu.csv"
combined.to_csv(combined_path, index=False)
print(f"Combined dataset saved as '{combined_path}'")

# Binary label encoding (Benign=0, Malicious=1)
combined['label'] = combined['label'].astype(str).apply(
    lambda x: 0 if 'BENIGN' in x.upper() or 'NORMAL' in x.upper() else 1
)

# Ensure numeric
for col in combined.columns:
    if col not in ['label', 'protocol']:
        combined[col] = pd.to_numeric(combined[col], errors='coerce').fillna(0)

# Encode protocol as categorical codes
combined['protocol'] = combined['protocol'].astype('category').cat.codes

print("\nCleaned dataset summary:")
print(combined['label'].value_counts())

# Train Baseline Gradient Boosting Model

X = combined.drop(columns=['label'])
y = combined['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, target_names=['Benign', 'Malicious'])
roc_auc = roc_auc_score(y_test, y_prob)

print("\nMODEL TRAINED SUCCESSFULLY")
print(report)
print(f"ROC-AUC: {roc_auc:.3f}")

joblib.dump(model, 'baseline_mal.pkl')
with open('metrics_mal.txt', 'w') as f:
    f.write(report)
    f.write(f"\nROC-AUC: {roc_auc:.3f}\n")

print("Model saved as 'baseline_mal.pkl' and metrics in 'metrics_mal.txt'")
