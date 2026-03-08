# isoforest_per_protocol.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
INPUT = "combined_cic_ctu.csv"
OUT_SCORES = "if_scores_per_protocol.csv"
MODELS_DIR = "if_models"
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
BENIGN_TRAIN_MAX = 200_000     # per protocol max benign training samples
MIN_BENIGN_FOR_MODEL = 2000    # only train per-protocol model if >= this many benign samples
IF_N_EST = 300
IF_MAX_SAMPLES = 50000
IF_CONTAMINATION = 0.005       # initial, we'll threshold on PR later

# ---------- HELPER ----------
def normalize_label_str(s):
    if pd.isna(s):
        return s
    s2 = str(s).strip()
    return s2

def is_exact_benign(s):
    if pd.isna(s):
        return False
    s2 = str(s).strip().upper()
    # strict equality to BENIGN
    return s2 == "BENIGN"

# ---------- LOAD ----------
print("Loading dataset...")
df_raw = pd.read_csv(INPUT, low_memory=False)
print("Rows loaded:", len(df_raw))

# Keep original label strings for strict benign selection
df_raw['label_orig'] = df_raw.get('label', "").astype(str)

# Strict benign mask: label exactly 'BENIGN' (case-insensitive)
mask_benign_strict = df_raw['label_orig'].str.strip().str.upper() == "BENIGN"
num_strict_benign = mask_benign_strict.sum()
print("Strict BENIGN rows:", num_strict_benign)

# If strict BENIGN is too few, you should re-evaluate label source.
if num_strict_benign < 1000:
    print("WARNING: Very few strict BENIGN rows. Consider checking original labels.")

# ---------- Select and clean features ----------
# Choose stable features (present in both CIC & CTU variants)
wanted = [
    'duration','flow_iat_mean','flow_iat_std',
    'fwd_pkts','bwd_pkts','fwd_bytes','bwd_bytes','protocol'
]
# ensure columns exist (create zeros if missing)
for c in wanted:
    if c not in df_raw.columns:
        df_raw[c] = 0

df = df_raw[wanted + ['label_orig']].copy()
# numeric conversion
for c in wanted:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

# ---------- Engineered features (compact and robust) ----------
eps = 1e-9
df['tot_pkts'] = df['fwd_pkts'] + df['bwd_pkts']
df['tot_bytes'] = df['fwd_bytes'] + df['bwd_bytes']
df['pkt_rate'] = df['tot_pkts'] / (df['duration'] + eps)
df['byte_rate'] = df['tot_bytes'] / (df['duration'] + eps)
df['fwd_bwd_ratio'] = (df['fwd_pkts'] + 1) / (df['bwd_pkts'] + 1)
df['avg_pkt_size'] = df['tot_bytes'] / (df['tot_pkts'] + eps)
# log transforms
df['log_pkt_rate'] = np.log1p(df['pkt_rate'].abs())
df['log_byte_rate'] = np.log1p(df['byte_rate'].abs())
df['log_avg_pkt_size'] = np.log1p(df['avg_pkt_size'].abs())

feature_cols = [
    'duration','flow_iat_mean','flow_iat_std',
    'tot_pkts','tot_bytes','pkt_rate','byte_rate',
    'fwd_bwd_ratio','avg_pkt_size',
    'log_pkt_rate','log_byte_rate','log_avg_pkt_size','protocol'
]
# ensure final dtype numeric
df[feature_cols] = df[feature_cols].astype(float)

# ---------- Per-protocol modeling ----------
# use protocol numeric code; if protocols were non-numeric earlier, they were coerce->0.
proto_counts = df['protocol'].value_counts()
print("Top protocols (count):")
print(proto_counts.head(10))

# list of protocols to train separate IF models on
protocols_to_train = [int(p) for p, cnt in proto_counts.items() if cnt >= 5000]  # adjust threshold
print("Protocols with enough data:", protocols_to_train)

# Prepare storage for metrics
metrics_lines = []
global_scores = []

# Iterate protocols: train per-protocol IF on strict benign subset for that protocol
for proto in protocols_to_train:
    proto_mask = (df['protocol'] == proto)
    proto_df = df[proto_mask].copy()
    # strict benign rows for this protocol
    proto_benign_mask = proto_df['label_orig'].str.strip().str.upper() == "BENIGN"
    num_benign = proto_benign_mask.sum()
    print(f"\nProtocol {proto}: rows={len(proto_df)}, strict_benign={num_benign}")
    if num_benign < MIN_BENIGN_FOR_MODEL:
        print("Skipping protocol due to too few strict benign rows.")
        continue

    # sample benign training set (cover diversity)
    benign_proto_df = proto_df[proto_benign_mask]
    n_train = min(BENIGN_TRAIN_MAX, len(benign_proto_df))
    benign_train = benign_proto_df.sample(n=n_train, random_state=RANDOM_STATE)

    X_train = benign_train[feature_cols].values
    # remove degenerate columns (zero variance) before scaling
    var = X_train.var(axis=0)
    keep_idx = np.where(var > 1e-8)[0]
    if len(keep_idx) < len(feature_cols):
        cols_kept = [feature_cols[i] for i in keep_idx]
        print("Dropping zero-variance features for proto", proto, "-> kept:", cols_kept)
    else:
        cols_kept = feature_cols.copy()

    # scale per-protocol
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(benign_train[cols_kept].values)
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"scaler_proto_{proto}.pkl"))

    # Train IF
    iso = IsolationForest(
        n_estimators=IF_N_EST,
        max_samples=IF_MAX_SAMPLES if IF_MAX_SAMPLES != 'auto' else 'auto',
        contamination=IF_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    iso.fit(X_train_scaled)
    joblib.dump(iso, os.path.join(MODELS_DIR, f"if_model_proto_{proto}.pkl"))
    print("Saved model for proto", proto)

    # Score all flows for this protocol
    X_all_proto = proto_df[cols_kept].values
    X_all_scaled = scaler.transform(X_all_proto)
    raw_scores = iso.decision_function(X_all_scaled)
    anomaly_score = -raw_scores
    pred_if = (iso.predict(X_all_scaled) == -1).astype(int)

    # Attach scores back to global dataframe via index
    proto_indices = proto_df.index.to_numpy()
    df.loc[proto_indices, f'if_score_p{proto}'] = anomaly_score
    df.loc[proto_indices, f'if_pred_p{proto}'] = pred_if

    # Evaluate for this protocol (use label_orig to evaluate)
    y_true_proto = (proto_df['label_orig'].str.strip().str.upper() != "BENIGN").astype(int).values
    # compute PR AUC for scores
    try:
        prec, rec, thr = precision_recall_curve(y_true_proto, anomaly_score)
        pr_auc = auc(rec, prec)
    except Exception as e:
        pr_auc = np.nan
    cm = confusion_matrix(y_true_proto, pred_if)
    report = classification_report(y_true_proto, pred_if, output_dict=False)
    metrics_lines.append(f"=== Protocol {proto} | rows={len(proto_df)} | benign_train={n_train} ===\n")
    metrics_lines.append(f"Confusion matrix (pred_if):\n{cm}\n")
    metrics_lines.append(f"PR-AUC (score): {pr_auc:.6f}\n")
    metrics_lines.append(f"{report}\n\n")

    # Save kde plot for protocol
    plt.figure(figsize=(6,4))
    sns.kdeplot(pd.Series(anomaly_score)[y_true_proto==0], label='Benign', bw_adjust=1)
    sns.kdeplot(pd.Series(anomaly_score)[y_true_proto==1], label='Malicious', bw_adjust=1)
    plt.title(f"Proto {proto} anomaly score KDE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f"if_kde_proto_{proto}.png"))
    plt.close()

# ---------- Global fallback model (for small/unknown protocols) ----------
# Train a fallback global IF on strict benign rows across all protocols
print("\nTraining fallback global IF on strict benign rows across all protocols...")
strict_benign_df = df[df['label_orig'].str.strip().str.upper() == "BENIGN"].copy()
if len(strict_benign_df) < 1000:
    print("Not enough strict benign rows for global fallback; aborting.")
else:
    n_train = min(BENIGN_TRAIN_MAX, len(strict_benign_df))
    benign_train = strict_benign_df.sample(n=n_train, random_state=RANDOM_STATE)
    cols_kept = feature_cols.copy()
    # drop zero variance features
    var = benign_train[cols_kept].var(axis=0)
    keep_idx = np.where(var > 1e-8)[0]
    cols_kept = [cols_kept[i] for i in keep_idx]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(benign_train[cols_kept].values)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_fallback.pkl"))

    iso = IsolationForest(n_estimators=IF_N_EST, max_samples=IF_MAX_SAMPLES, contamination=IF_CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train_scaled)
    joblib.dump(iso, os.path.join(MODELS_DIR, "if_model_fallback.pkl"))
    print("Saved fallback model.")

    # Score all rows not covered by per-protocol models
    # For rows where per-protocol score exists, we'll keep that; else apply fallback
    covered = df[[c for c in df.columns if c.startswith('if_score_p')]].notna().any(axis=1)
    to_score_idx = df[~covered].index
    if len(to_score_idx) > 0:
        X_to_score = df.loc[to_score_idx, cols_kept].values
        X_to_score_scaled = scaler.transform(X_to_score)
        raw_scores = iso.decision_function(X_to_score_scaled)
        anomaly_score = -raw_scores
        pred_if = (iso.predict(X_to_score_scaled) == -1).astype(int)
        df.loc[to_score_idx, 'if_score_fallback'] = anomaly_score
        df.loc[to_score_idx, 'if_pred_fallback'] = pred_if

# ---------- Finalize scoring columns: collapse per-proto and fallback into single score/pred ----------
score_cols = [c for c in df.columns if c.startswith('if_score_')]
pred_cols = [c for c in df.columns if c.startswith('if_pred_')]

# Collapse: prefer per-protocol score if exists, else fallback
df['if_score_final'] = np.nan
df['if_pred_final'] = np.nan
for idx in df.index:
    # find first available score/pred in row
    found = False
    for sc in score_cols:
        v = df.at[idx, sc]
        if pd.notna(v):
            df.at[idx, 'if_score_final'] = v
            pred_name = sc.replace('score','pred')
            df.at[idx, 'if_pred_final'] = df.at[idx, pred_name]
            found = True
            break
    if not found:
        # fallback columns
        if 'if_score_fallback' in df.columns and pd.notna(df.at[idx, 'if_score_fallback']):
            df.at[idx, 'if_score_final'] = df.at[idx, 'if_score_fallback']
            df.at[idx, 'if_pred_final'] = df.at[idx, 'if_pred_fallback']
        else:
            df.at[idx, 'if_score_final'] = np.nan
            df.at[idx, 'if_pred_final'] = np.nan

# ---------- Evaluate globally using label_orig (benign=0, malicious=1) ----------
df['label_bin'] = (df['label_orig'].str.strip().str.upper() != "BENIGN").astype(int)
valid_idx = df['if_pred_final'].notna()
y_true = df.loc[valid_idx, 'label_bin'].values
y_pred = df.loc[valid_idx, 'if_pred_final'].astype(int).values
y_score = df.loc[valid_idx, 'if_score_final'].astype(float).values

print("\nGlobal evaluation (on rows covered by models):")
print("Rows covered:", valid_idx.sum())
print("Confusion matrix (final):")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

try:
    print("Global ROC-AUC (score):", roc_auc_score(y_true, y_score))
except Exception as e:
    print("ROC-AUC error:", e)

# Save metrics and scores
with open(os.path.join(MODELS_DIR, "if_metrics_per_protocol.txt"), "w") as mf:
    mf.write("Per-protocol metrics:\n\n")
    mf.write("\n".join(metrics_lines))
    mf.write("\nGLOBAL evaluation:\n")
    mf.write("Rows covered: %d\n" % valid_idx.sum())
    mf.write("Confusion matrix:\n")
    mf.write(str(confusion_matrix(y_true, y_pred)) + "\n")
    mf.write("\nClassification report:\n")
    mf.write(classification_report(y_true, y_pred))
    try:
        mf.write("\nROC-AUC (score): %.6f\n" % roc_auc_score(y_true, y_score))
    except:
        mf.write("\nROC-AUC (score): N/A\n")

df_out = df[feature_cols + ['label_orig', 'label_bin', 'if_score_final', 'if_pred_final']]
df_out.to_csv(OUT_SCORES, index=False)
print("Saved scores to", OUT_SCORES)
print("Per-protocol models in", MODELS_DIR)
print("Done.")
