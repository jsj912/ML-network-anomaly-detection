# compute_if_explain_fixed.py
import os, json, time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
SCORES_CSV = "two_stage_scores.csv"
IF_PKL = "if_stage.pkl"   # make sure this is the IF model YOU saved
SCALER_PKL = "if_scaler.pkl"         # scaler file you saved during IF training
FEATURES_JOBLIB = "if_feature_cols.joblib"  # optional: list of features used during IF training
OUT_DIR = "if_explain"
TOP_N = 2000
CHUNK = 300_000

Path(OUT_DIR).mkdir(exist_ok=True)

# ---------- helpers ----------
def read_top_n_scores(n=TOP_N):
    if not os.path.exists(SCORES_CSV):
        raise FileNotFoundError(SCORES_CSV)
    top_df = None
    reader = pd.read_csv(SCORES_CSV, chunksize=CHUNK, engine='python', on_bad_lines='skip')
    for chunk in reader:
        if 'rerank_score' not in chunk.columns:
            return chunk.head(n)
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

def sample_benign_values(samples=5000):
    # sample benign rows to compute medians / means
    benign_parts = []
    reader = pd.read_csv(SCORES_CSV, chunksize=CHUNK, engine='python', on_bad_lines='skip')
    for chunk in reader:
        if 'label' not in chunk.columns:
            continue
        # treat label case-insensitive
        mask = chunk['label'].astype(str).str.upper() == "BENIGN"
        if mask.any():
            s = chunk.loc[mask, :]
            # sample up to min(1000, len(s))
            n = min(1000, len(s))
            benign_parts.append(s.sample(n=n, random_state=42))
        if sum(len(x) for x in benign_parts) >= samples:
            break
    if len(benign_parts) == 0:
        return None
    bdf = pd.concat(benign_parts, ignore_index=True)
    return bdf

# ---------- main ----------
print("Loading IF model & scaler")
if not os.path.exists(IF_PKL) or not os.path.exists(SCALER_PKL):
    raise FileNotFoundError("IF model or scaler not found. Ensure you have isolation_forest_scaled.pkl and scaler_robust.pkl")

iso = joblib.load(IF_PKL)
scaler = joblib.load(SCALER_PKL)

print("Reading top candidates...")
top_df = read_top_n_scores(TOP_N)
print("Top candidates:", len(top_df))

# infer features present in top_df (numeric ones)
exclude = {'label','label_bin','if_score','if_flag','rerank_score','rerank_pred','top_index'}
data_feat_cols = [c for c in top_df.columns if c not in exclude and top_df[c].dtype.kind in 'fi']
print("Numeric columns found in top candidates:", len(data_feat_cols))

# Determine scaler-expected features
scaler_features = None
if hasattr(scaler, "feature_names_in_"):
    scaler_features = list(scaler.feature_names_in_)
    print("Scaler exposes feature_names_in_ with length:", len(scaler_features))
elif os.path.exists(FEATURES_JOBLIB):
    try:
        scaler_features = joblib.load(FEATURES_JOBLIB)
        print("Loaded feature list from", FEATURES_JOBLIB, "len:", len(scaler_features))
    except Exception:
        scaler_features = None

if scaler_features is None:
    print("Warning: scaler does not expose feature names and no joblib feature list found.")
    print("Proceeding with intersection of available numeric features (may be approximate).")
    scaler_features = data_feat_cols.copy()  # fallback

# Now compute final feature list to feed to scaler:
# - For any feature expected by scaler but missing in data, we will fill with benign median (if available) or 0.
# - For any feature present in data but not expected by scaler, we drop it.
final_features = scaler_features.copy()
print("Final features to use (len):", len(final_features))

# sample benign rows to compute medians for missing features
print("Sampling benign rows to compute medians for missing features...")
benign_df = sample_benign_values(samples=5000)
if benign_df is None:
    print("Warning: no benign rows found in scores CSV; missing-feature fill will use zeros.")
else:
    print("Benign sample rows:", len(benign_df))

# Build matrix X aligned to final_features
X_rows = []
for idx, row in top_df.iterrows():
    vals = []
    for f in final_features:
        if f in top_df.columns and pd.notna(row[f]):
            try:
                vals.append(float(row[f]))
            except Exception:
                vals.append(0.0)
        else:
            # missing feature: fill with benign median if available else 0
            if benign_df is not None and f in benign_df.columns:
                med = float(pd.to_numeric(benign_df[f], errors='coerce').median(skipna=True))
                vals.append(med)
            else:
                vals.append(0.0)
    X_rows.append(vals)

X = np.array(X_rows, dtype=float)
print("Constructed X with shape:", X.shape)

# Check scaler expected n_features
try:
    expected = scaler.n_features_in_
    if X.shape[1] != expected:
        # If scaler.feature_names_in_ exists, we already used it; otherwise try to handle mismatch
        print(f"Scaler expects {expected} features but X has {X.shape[1]}. Attempting to reconcile...")

        # If X has fewer features, append zeros columns
        if X.shape[1] < expected:
            diff = expected - X.shape[1]
            print(f"Appending {diff} zero-columns to match scaler")
            X = np.hstack([X, np.zeros((X.shape[0], diff), dtype=float)])
        else:
            # X has more features than scaler expects: truncate from end
            print(f"Truncating X from {X.shape[1]} to {expected}")
            X = X[:, :expected]
except Exception:
    pass

# Transform using scaler
print("Scaling X with the loaded scaler...")
X_scaled = scaler.transform(X)
print("X_scaled shape:", X_scaled.shape)

# Baseline IF scores (if decision_function used earlier)
base_scores = -iso.decision_function(X_scaled)   # higher -> more anomalous
print("Computed base IF scores for top candidates")

# -------------- Ablation: for each feature, replace with benign median (in original feature space),
# compute new IF score and get delta --------------
print("Computing benign medians for final_features (for ablation)...")
if benign_df is not None:
    benign_medians = {}
    for f in final_features:
        if f in benign_df.columns:
            benign_medians[f] = float(pd.to_numeric(benign_df[f], errors='coerce').median(skipna=True))
        else:
            benign_medians[f] = 0.0
else:
    benign_medians = {f: 0.0 for f in final_features}

# transform benign medians into scaled space in bulk
median_row = np.array([[benign_medians[f] for f in final_features]], dtype=float)
median_scaled = scaler.transform(median_row)[0]  # shape (n_features,)

ablation_results = []
for i in range(X_scaled.shape[0]):
    orig = X_scaled[i].copy()
    deltas = []
    # restore original unscaled row to modify individual features in original space conveniently:
    # We can inverse_transform the scaled vector to get approximate original (if scaler supports inverse)
    try:
        orig_unscaled = scaler.inverse_transform(orig.reshape(1, -1))[0]
    except Exception:
        # fallback: use median replacement in scaled space directly using median_scaled
        orig_unscaled = None

    for j, f in enumerate(final_features):
        if orig_unscaled is not None:
            # replace the j-th original value with benign median and scale
            tmp = orig_unscaled.copy()
            tmp[j] = benign_medians[f]
            tmp_scaled = scaler.transform(tmp.reshape(1, -1))[0]
        else:
            # direct scaled replacement: replace scaled entry with median_scaled
            tmp_scaled = orig.copy()
            tmp_scaled[j] = median_scaled[j]

        s = -iso.decision_function([tmp_scaled])[0]
        delta = float(s - base_scores[i])
        deltas.append((f, delta))
    # sort by absolute delta descending
    deltas_sorted = sorted(deltas, key=lambda x: abs(x[1]), reverse=True)[:10]
    ablation_results.append({"top_index": int(i), "base_score": float(base_scores[i]), "deltas": [{"feature": d[0], "delta": d[1]} for d in deltas_sorted]})

# -------------- Z-score distance from benign centroid (in original feature space) --------------
print("Computing z-scores (distance from benign centroid)...")
if benign_df is not None:
    benign_mean = benign_df[final_features].astype(float).mean()
    benign_std = benign_df[final_features].astype(float).std().replace(0, 1e-9)
else:
    benign_mean = pd.Series({f: 0.0 for f in final_features})
    benign_std = pd.Series({f: 1.0 for f in final_features})

z_scores = []
for i in range(X.shape[0]):
    orig_vals = X[i]
    # orig_vals correspond to final_features order already
    z = np.abs((orig_vals - benign_mean.values) / benign_std.values)
    # build top features by z
    idxs = np.argsort(-z)[:10]
    topz = [{"feature": final_features[k], "z": float(z[k])} for k in idxs]
    z_scores.append({"top_index": int(i), "top_z": topz})

# -------------- LOF --------------
print("Computing LOF on top candidates...")
lof = LocalOutlierFactor(n_neighbors=20, novelty=False, metric='minkowski', n_jobs=-1)
# LOF expects 2D array; we use scaled features
try:
    lof_pred = lof.fit_predict(X_scaled)
    lof_scores = -lof.negative_outlier_factor_
except Exception as e:
    print("LOF failed:", e)
    lof_scores = np.zeros(X_scaled.shape[0])

# -------------- Save outputs --------------
print("Saving outputs to", OUT_DIR)
with open(os.path.join(OUT_DIR, "if_ablation_topN.json"), "w") as f:
    json.dump(ablation_results, f, indent=2)

with open(os.path.join(OUT_DIR, "if_zscores_topN.json"), "w") as f:
    json.dump(z_scores, f, indent=2)

pd.DataFrame({"top_index": list(range(len(lof_scores))), "lof_score": list(map(float, lof_scores))}).to_csv(os.path.join(OUT_DIR, "if_lof_topN.csv"), index=False)

# combined summary CSV for dashboard
summary_rows = []
for i in range(X_scaled.shape[0]):
    row = top_df.iloc[i]
    summary_rows.append({
        "top_index": int(i),
        "rerank_score": float(row.get("rerank_score", 0)),
        "if_score": float(base_scores[i]),
        "lof_score": float(lof_scores[i]),
        "label_bin": int(row.get("label_bin", 0))
    })
pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "if_aggregate_topN.csv"), index=False)

print("Saved IF explain outputs to", OUT_DIR)
print("Done.")
