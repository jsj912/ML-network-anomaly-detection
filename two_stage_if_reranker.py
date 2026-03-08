# two_stage_if_reranker_fixed.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, 
    precision_recall_curve, auc, confusion_matrix
)
import joblib
from lightgbm import LGBMClassifier, early_stopping
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
INPUT = "combined_cic_ctu.csv"
OUT_SCORES = "two_stage_scores.csv"

IF_MODEL = "if_stage.pkl"
IF_SCALER = "if_scaler.pkl"
RERANKER_MODEL = "reranker_lgbm.pkl"

RANDOM_STATE = 42

IF_N_EST = 300
IF_MAX_SAMPLES = 50000
IF_CONTAMINATION = 0.015

TOP_K_IF = 350000       # Capture more anomalies
NEG_SAMPLES = 350000    # More negatives for reranker

# ----------- STEP 1: LOAD DATA -----------
df = pd.read_csv(INPUT, low_memory=False)
print("Loaded rows:", len(df))

# Normalize label
df["label_bin"] = df["label"].astype(str).apply(
    lambda x: 0 if x.strip().upper() == "BENIGN" else 1
)

# ----------- STEP 2: PROTOCOL ENCODING -----------
proto_map = {
    "tcp": 6,
    "udp": 17,
    "icmp": 1,
    "rtp": 5004,
    "rtcp": 5005,
    "arp": 2054,
    "ipv6-icmp": 58,
    "igmp": 2,
    "esp": 50,
    "gre": 47,
    "rarp": 0,
}

def encode_proto(x):
    x = str(x).lower()
    if x.isdigit():
        return int(x)
    return proto_map.get(x, 0)

df["protocol"] = df["protocol"].apply(encode_proto)

# ----------- STEP 3: FEATURE ENGINEERING -----------
eps = 1e-9

df["tot_pkts"] = df["fwd_pkts"] + df["bwd_pkts"]
df["tot_bytes"] = df["fwd_bytes"] + df["bwd_bytes"]
df["pkt_rate"] = df["tot_pkts"] / (df["duration"] + eps)
df["byte_rate"] = df["tot_bytes"] / (df["duration"] + eps)
df["fwd_bwd_ratio"] = (df["fwd_pkts"] + 1) / (df["bwd_pkts"] + 1)
df["avg_pkt_size"] = df["tot_bytes"] / (df["tot_pkts"] + eps)

df["log_pkt_rate"] = np.log1p(df["pkt_rate"])
df["log_byte_rate"] = np.log1p(df["byte_rate"])
df["log_avg_pkt_size"] = np.log1p(df["avg_pkt_size"])

feature_cols = [
    "duration", "flow_iat_mean", "flow_iat_std",
    "tot_pkts", "tot_bytes", "pkt_rate",
    "byte_rate", "avg_pkt_size",
    "fwd_bwd_ratio", "protocol",
    "log_pkt_rate", "log_byte_rate", "log_avg_pkt_size"
]

df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# ----------- STEP 4: TRAIN IF ON STRICT BENIGN -----------
strict_benign = df[df["label_bin"] == 0]

print("Strict benign count:", len(strict_benign))

train_benign = strict_benign[feature_cols].sample(
    n=min(300000, len(strict_benign)),
    random_state=RANDOM_STATE
)

scaler = RobustScaler()
Xb = scaler.fit_transform(train_benign)
joblib.dump(scaler, IF_SCALER)

iso = IsolationForest(
    n_estimators=IF_N_EST,
    max_samples=IF_MAX_SAMPLES,
    contamination=IF_CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
iso.fit(Xb)
joblib.dump(iso, IF_MODEL)

# ----------- STEP 5: SCORE ALL WITH IF -----------
X_all = scaler.transform(df[feature_cols])
df["if_score"] = -iso.decision_function(X_all)
df["if_flag"] = (iso.predict(X_all) == -1).astype(int)

# ----------- STEP 6: BUILD RERANKER TRAIN SET -----------
candidates = df.sort_values("if_score", ascending=False).head(TOP_K_IF)

pos = candidates[candidates["label_bin"] == 1]
neg = df[df["label_bin"] == 0].sample(NEG_SAMPLES, random_state=RANDOM_STATE)

train_df = pd.concat([pos, neg])
X = train_df[feature_cols].values
y = train_df["label_bin"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# ----------- STEP 7: TRAIN LIGHTGBM RERANKER -----------
lgb = LGBMClassifier(
    n_estimators=700,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE
)

lgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(stopping_rounds=50)]
)

joblib.dump(lgb, RERANKER_MODEL)
print("Saved reranker:", RERANKER_MODEL)

# ----------- STEP 8: FINAL SCORING + SAVE -----------
df["rerank_score"] = lgb.predict_proba(df[feature_cols])[:, 1]
df["rerank_pred"] = (df["rerank_score"] >= 0.5).astype(int)

df.to_csv(OUT_SCORES, index=False)
print("Saved:", OUT_SCORES)

# ----------- STEP 9: REPORT -----------
eval_df = df.sample(300000, random_state=RANDOM_STATE)
y_eval = eval_df["label_bin"]
y_score = eval_df["rerank_score"]
y_pred  = eval_df["rerank_pred"]

print("\nRERANKER REPORT:")
print(classification_report(y_eval, y_pred))

print("ROC-AUC:", roc_auc_score(y_eval, y_score))

# Precision@K
eval_sorted = eval_df.sort_values("rerank_score", ascending=False)
for K in [100, 500, 1000, 5000]:
    topk = eval_sorted.head(K)
    prec = topk["label_bin"].mean()
    print(f"Precision@{K}: {prec:.4f}")
