# compute_gbm_shap.py
import os, json, time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
SCORES_CSV = "two_stage_scores.csv"
GBM_PKL = "baseline_lightgbm.pkl"   # <-- change if your baseline model filename differs
OUT_DIR = "shap_gbm"
TOP_N = 1000                  # how many top alerts to explain
BG_SIZE = 20000               # background size for TreeExplainer (interventional)
CHUNK = 300_000
FEATURE_COLS_DEFAULT = [
    "duration","fwd_pkts","bwd_pkts","fwd_bytes","bwd_bytes",
    "flow_iat_mean","flow_iat_std","protocol","tot_pkts","tot_bytes",
    "pkt_rate","byte_rate","fwd_bwd_ratio","avg_pkt_size",
    "log_pkt_rate","log_byte_rate","log_avg_pkt_size"
]
Path(OUT_DIR).mkdir(exist_ok=True)

# ---------- helpers ----------
def read_top_n_scores(n=TOP_N):
    if not os.path.exists(SCORES_CSV):
        raise FileNotFoundError(SCORES_CSV)
    top_df = None
    reader = pd.read_csv(SCORES_CSV, chunksize=CHUNK, engine='python', on_bad_lines='skip')
    for chunk in reader:
        if 'rerank_score' not in chunk.columns:
            # fallback to simplest head
            return chunk.head(n).reset_index(drop=True)
        chunk['rerank_score'] = pd.to_numeric(chunk['rerank_score'], errors='coerce').fillna(0)
        chunk = chunk.dropna(subset=['rerank_score'])
        if top_df is None:
            top_df = chunk.nlargest(n, 'rerank_score')
        else:
            combined = pd.concat([top_df, chunk], ignore_index=True)
            top_df = combined.nlargest(n, 'rerank_score')
    if top_df is None:
        return pd.DataFrame()
    top_df = top_df.sort_values('rerank_score', ascending=False).reset_index(drop=True)
    top_df['top_index'] = top_df.index
    return top_df

# ---------- main ----------
start = time.time()
print("Loading model:", GBM_PKL)
model = joblib.load(GBM_PKL)

print("Reading top alerts from scores (this scans CSV in chunks):")
top_df = read_top_n_scores(TOP_N)
print("Top rows:", len(top_df))

# Choose feature columns: prefer explicit defaults, but take what's available in CSV
available = [c for c in FEATURE_COLS_DEFAULT if c in top_df.columns]
if not available:
    # fallback: take numeric columns except label-like columns
    available = [c for c in top_df.select_dtypes(include=['number']).columns if c not in ['label','label_bin','rerank_score','top_index','if_score','if_flag']]
print("Using feature columns:", available)

# build background from the rest of the CSV (sample)
print("Building background sample of size", BG_SIZE)
# sample from the end of the CSV by streaming if needed
bg_rows = []
reader = pd.read_csv(SCORES_CSV, chunksize=CHUNK, engine='python', on_bad_lines='skip')
for chunk in reader:
    # pick random subset from chunk
    if len(chunk) <= 0: 
        continue
    sample = chunk[available].sample(n=min(int(BG_SIZE/10), len(chunk)), random_state=42)
    bg_rows.append(sample)
    if sum(len(x) for x in bg_rows) >= BG_SIZE:
        break
if len(bg_rows) == 0:
    raise RuntimeError("Failed to build background.")
bg_df = pd.concat(bg_rows, ignore_index=True).sample(n=min(BG_SIZE, sum(len(x) for x in bg_rows)), random_state=42)
X_bg = bg_df[available].astype(float).values
X_top = top_df[available].astype(float).values

# ---------- SHAP explainer (TreeExplainer interventional) ----------
print("Creating TreeExplainer (interventional mode). This may take some memory/time.")
explainer = shap.TreeExplainer(model, data=X_bg, feature_perturbation="interventional")
joblib.dump(explainer, os.path.join(OUT_DIR, "shap_gbm_explainer.joblib"))

print("Computing SHAP values for top N...")
raw_sv = explainer.shap_values(X_top, check_additivity=False)
# LightGBM returns list for binary; use positive class
if isinstance(raw_sv, list):
    shap_vals = raw_sv[1] if len(raw_sv) > 1 else raw_sv[0]
else:
    shap_vals = raw_sv
print("SHAP shape:", shap_vals.shape)

# Save arrays for fast reloading
np.save(os.path.join(OUT_DIR, "gbm_shap_values.npy"), shap_vals)
joblib.dump(available, os.path.join(OUT_DIR, "gbm_feature_cols.joblib"))

# ---------- Global SHAP summary (use smaller sample to make plot) ----------
print("Generating global summary plot (beeswarm)...")
# sample rows for global summary (from bg + top)
X_summary = np.vstack([X_bg[:min(5000, len(X_bg))], X_top[:min(5000, len(X_top))]])
sv_summary = explainer.shap_values(X_summary, check_additivity=False)
if isinstance(sv_summary, list):
    sv_summary = sv_summary[1] if len(sv_summary) > 1 else sv_summary[0]
shap.summary_plot(sv_summary, features=X_summary, feature_names=available, show=False, plot_size=(10,6))
plt.savefig(os.path.join(OUT_DIR, "global_shap_summary.png"), bbox_inches='tight', dpi=150)
plt.close()

# ---------- Per-alert JSON + PNG waterfall plots ----------
print("Writing per-alert JSON and PNGs...")
explanations = []
for i in range(len(top_df)):
    row = top_df.iloc[i]
    sv = shap_vals[i]
    pairs = [{"feature": available[j], "shap_value": float(sv[j]), "value": float(row[available[j]]) if available[j] in row else None} for j in range(len(available))]
    pairs_sorted = sorted(pairs, key=lambda x: abs(x["shap_value"]), reverse=True)
    explanations.append({
        "top_index": int(i),
        "global_index": int(row.name),
        "rerank_score": float(row.get("rerank_score", 0)),
        "label_bin": int(row.get("label_bin", 0)),
        "top_contributors": pairs_sorted[:10]
    })
    # PNG bar plot of top contributors
    nshow = min(12, len(available))
    idxs = np.argsort(-np.abs(sv))[:nshow]
    names = [available[k] for k in idxs]
    vals = sv[idxs]
    colors = ['red' if v>0 else 'blue' for v in vals]
    plt.figure(figsize=(8,4))
    plt.barh(range(len(vals))[::-1], vals[::-1], color=colors[::-1])
    plt.yticks(range(len(vals))[::-1], [names[::-1][k] for k in range(len(names))])
    plt.title(f"GBM SHAP alert #{i} (score={row.get('rerank_score',0):.4f})")
    plt.xlabel("SHAP value")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"gbm_shap_alert_{i}.png"), dpi=150)
    plt.close()

with open(os.path.join(OUT_DIR, "gbm_shap_topN_explanations.json"), "w") as f:
    json.dump(explanations, f, indent=2)

print("Saved GBM SHAP outputs to", OUT_DIR)
print("Elapsed:", time.time()-start)
