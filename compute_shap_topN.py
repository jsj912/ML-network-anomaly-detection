# compute_shap_topN_fixed.py
import os, json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

# -------- CONFIG --------
TOP_N = 1000
SCORES_CSV = "two_stage_scores.csv"
RERANKER_PKL = "reranker_lgbm.pkl"
OUTPUT_DIR = "shap_topN"

# Larger background sample (fixes TreeExplainer error)
BACKGROUND_SIZE = 20000

# -------- LOAD DATA & MODEL --------
print("Loading scores:", SCORES_CSV)
df = pd.read_csv(SCORES_CSV, low_memory=False)
print("Rows:", len(df))

print("Loading reranker:", RERANKER_PKL)
model = joblib.load(RERANKER_PKL)

# Detect feature columns automatically
drop_cols = {'label','label_bin','if_score','if_flag','rerank_score','rerank_pred'}
feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype.kind in "fi"]

print("Using features:", feature_cols)
print("Count:", len(feature_cols))

# -------- SELECT TOP-N --------
df_sorted = df.sort_values("rerank_score", ascending=False)
top_df = df_sorted.head(TOP_N).reset_index(drop=True)
X_top = top_df[feature_cols].values

# -------- CREATE LARGE BACKGROUND --------
print("Creating background sample...")

if len(df) > BACKGROUND_SIZE:
    bg_df = df.sample(BACKGROUND_SIZE, random_state=42)
else:
    bg_df = df.copy()

X_bg = bg_df[feature_cols].values

print("Background size:", X_bg.shape)

# -------- CREATE EXPLAINER (INTERVENTIONAL) --------
print("Creating SHAP TreeExplainer (interventional mode)...")

explainer = shap.TreeExplainer(
    model,
    data=X_bg,
    feature_perturbation="interventional"
)

joblib.dump(explainer, "shap_explainer.joblib")
print("Saved explainer: shap_explainer.joblib")

# -------- COMPUTE SHAP VALUES --------
print("Computing SHAP values... this may take time...")

# LightGBM binary -> shap returns list: [class0, class1]
raw_sv = explainer.shap_values(X_top, check_additivity=False)

if isinstance(raw_sv, list):
    shap_values = raw_sv[1]  # positive class
else:
    shap_values = raw_sv

print("SHAP shape:", shap_values.shape)

# -------- SAVE RAW SHAP DATA --------
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "shap_values.npy"), shap_values)
np.save(os.path.join(OUTPUT_DIR, "top_indices.npy"), top_df.index.values)
joblib.dump(feature_cols, os.path.join(OUTPUT_DIR, "feature_cols.joblib"))

# -------- GENERATE JSON + PER-ALERT PNGs --------
summary_json = []

for i in range(TOP_N):
    row = top_df.iloc[i]
    sv = shap_values[i]

    # Produce Top Contributors
    pairs = [
        {"feature": f, "shap_value": float(sv[j]), "value": float(row[f])}
        for j, f in enumerate(feature_cols)
    ]
    pairs_sorted = sorted(pairs, key=lambda x: abs(x["shap_value"]), reverse=True)
    top_feats = pairs_sorted[:10]

    summary_json.append({
        "local_index": i,
        "global_index": int(row.name),
        "rerank_score": float(row["rerank_score"]),
        "label_bin": int(row["label_bin"]),
        "top_contributors": top_feats
    })

    # Draw simple bar SHAP plot
    plt.figure(figsize=(8, 4))
    idxs = np.argsort(-np.abs(sv))[:12]  # top-12
    names = [feature_cols[k] for k in idxs]
    values = sv[idxs]
    colors = ["red" if v > 0 else "blue" for v in values]

    plt.barh(range(len(values))[::-1], values[::-1], color=colors[::-1])
    plt.yticks(range(len(values))[::-1], names[::-1])
    plt.title(f"Alert #{i} (score={row['rerank_score']:.4f})")
    plt.xlabel("SHAP value")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_alert_{i}.png"), dpi=150)
    plt.close()

# Save JSON
with open(os.path.join(OUTPUT_DIR, "shap_topN_explanations.json"), "w") as f:
    json.dump(summary_json, f, indent=2)

print("Saved SHAP images + JSON explanations in:", OUTPUT_DIR)
print("DONE.")
