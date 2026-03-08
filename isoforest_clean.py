import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================================
# CONFIG
# ===============================================================
INPUT = "combined_cic_ctu.csv"
OUTPUT_SCORES = "if_anomaly_scores.csv"
MODEL_PATH = "if_model.pkl"
SCALER_PATH = "if_scaler.pkl"
RANDOM_STATE = 42

# ===============================================================
# LOAD DATA
# ===============================================================
df = pd.read_csv(INPUT, low_memory=False)
print("Loaded:", df.shape)

# Normalize labels
def normalize_label(x):
    x = str(x).upper()
    if "BENIGN" in x or x == "0":
        return 0
    else:
        return 1

df["label"] = df["label"].apply(normalize_label)

# ===============================================================
# SELECT ONLY CLEAN FEATURES
# (These are stable and consistent across CIC + CTU)
# ===============================================================
keep = [
    "duration",
    "fwd_pkts",
    "bwd_pkts",
    "fwd_bytes",
    "bwd_bytes",
    "flow_iat_mean",
    "flow_iat_std",
    "protocol",
]

for col in keep:
    if col not in df.columns:
        df[col] = 0
        
df = df[keep + ["label"]]

# numeric convert
for col in keep:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ===============================================================
# FEATURE ENGINEERING (BUT CLEAN)
# ===============================================================
df["tot_pkts"] = df["fwd_pkts"] + df["bwd_pkts"]
df["tot_bytes"] = df["fwd_bytes"] + df["bwd_bytes"]

df["pkt_rate"] = df["tot_pkts"] / (df["duration"] + 1e-6)
df["byte_rate"] = df["tot_bytes"] / (df["duration"] + 1e-6)

df["fwd_bwd_ratio"] = (df["fwd_bytes"] + 1) / (df["bwd_bytes"] + 1)

df["avg_pkt_size"] = df["tot_bytes"] / (df["tot_pkts"] + 1e-6)

keep2 = [
    "duration","fwd_pkts","bwd_pkts","tot_pkts",
    "fwd_bytes","bwd_bytes","tot_bytes",
    "pkt_rate","byte_rate","avg_pkt_size",
    "fwd_bwd_ratio","protocol","flow_iat_mean","flow_iat_std",
]

df = df[keep2 + ["label"]]

print("Final feature matrix:", df.shape)

# ===============================================================
# CLEAN BENIGN TRAINING SET
# ===============================================================
benign = df[df["label"] == 0].copy()

# remove extremely small or large flows (garbage cleaning)
benign = benign[
    (benign["duration"] > 0) &
    (benign["tot_pkts"] > 0) &
    (benign["tot_bytes"] > 0)
]

# sample for training
benign_train = benign.sample(n=200000, random_state=RANDOM_STATE)
X_train = benign_train.drop(columns=["label"]).values

# ===============================================================
# SCALE FEATURES
# ===============================================================
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, SCALER_PATH)
print("Saved scaler")

X_all = df.drop(columns=["label"]).values
X_all_scaled = scaler.transform(X_all)

# ===============================================================
# TRAIN ISOLATION FOREST
# ===============================================================
iso = IsolationForest(
    n_estimators=400,
    max_samples=100000,
    contamination=0.005,     # <-- smaller contamination works better for CIC/CTU
    random_state=RANDOM_STATE,
    n_jobs=-1
)
iso.fit(X_train_scaled)
joblib.dump(iso, MODEL_PATH)
print("Saved IsolationForest model")

# ===============================================================
# SCORE ALL FLOWS
# ===============================================================
raw_scores = iso.decision_function(X_all_scaled)
anomaly_score = -raw_scores
y_true = df["label"].values

df["anomaly_score"] = anomaly_score
df["pred_if"] = (iso.predict(X_all_scaled) == -1).astype(int)

# ===============================================================
# THRESHOLD TUNING USING PRECISION-RECALL
# ===============================================================
prec, rec, thr = precision_recall_curve(y_true, anomaly_score)
f1 = 2 * (prec * rec) / (prec + rec + 1e-9)

best_idx = np.argmax(f1)
best_threshold = thr[best_idx]

df["pred_thresh"] = (df["anomaly_score"] >= best_threshold).astype(int)

# ===============================================================
# EVALUATE
# ===============================================================
print("\n---Isolation Forest (default decision function)---")
print(confusion_matrix(y_true, df["pred_if"]))
print(classification_report(y_true, df["pred_if"]))

print("\n---Isolation Forest (threshold tuned)---")
print("Best threshold:", best_threshold)
print(confusion_matrix(y_true, df["pred_thresh"]))
print(classification_report(y_true, df["pred_thresh"]))

try:
    print("ROC-AUC:", roc_auc_score(y_true, anomaly_score))
except:
    pass

# ===============================================================
# SAVE SCORES
# ===============================================================
df.to_csv(OUTPUT_SCORES, index=False)
print("Saved:", OUTPUT_SCORES)

# ===============================================================
# PLOTS
# ===============================================================
plt.figure(figsize=(8,5))
sns.kdeplot(df[df["label"]==0]["anomaly_score"], label="Benign")
sns.kdeplot(df[df["label"]==1]["anomaly_score"], label="Malicious")
plt.legend()
plt.title("Anomaly Score Distribution")
plt.savefig("if_kde.png")
print("Saved: if_kde.png")
