# make_anomaly_plot.py
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

OUTDIR = Path("eda_outputs")
OUTDIR.mkdir(exist_ok=True)

# prefer the merged combined_with_scores.csv, fallback to if_anomaly_scores.csv or two_stage_scores.csv
candidates = ["combined_with_scores.csv", "if_anomaly_scores.csv", "two_stage_scores.csv"]
df = None
score_col = None
label_col = None

for f in candidates:
    if Path(f).exists():
        print("Loading", f)
        # read a sample to find columns
        sample = pd.read_csv(f, nrows=5)
        cols = [c.lower() for c in sample.columns]
        # find score column
        for prefer in ['anomaly_score','if_score','rerank_score','score','if_score_final','if_score_final']:
            for c in sample.columns:
                if prefer == c.lower() or prefer in c.lower():
                    score_col = c
                    break
            if score_col:
                break
        # fallback numeric column
        if not score_col:
            for c in sample.columns:
                if sample[c].dtype.kind in 'fi':
                    score_col = c
                    break
        # label
        if 'label' in sample.columns:
            label_col = 'label'
        elif 'label_bin' in sample.columns:
            label_col = 'label_bin'
        # now load a reasonable chunk (or all if manageable)
        df = pd.read_csv(f, usecols=lambda c: c in (sample.columns) , low_memory=False)
        print("Loaded", f, "shape:", df.shape)
        break

if df is None or score_col is None:
    raise SystemExit("No score file found or no score column detected among candidates. Please run merge_scores_chunked.py first.")

print("Using score column:", score_col, "label column:", label_col)

# ensure numeric
df[score_col] = pd.to_numeric(df[score_col], errors='coerce')

# create label_bin numeric if label text exists
if label_col and df[label_col].dtype == object:
    df['label_bin'] = df[label_col].astype(str).str.upper().apply(lambda x: 0 if 'BENIGN' in x or 'NORMAL' in x else 1)
elif label_col:
    df['label_bin'] = df[label_col]
else:
    df['label_bin'] = np.nan

plt.figure(figsize=(10,6))
sns.set(style="whitegrid")
if df['label_bin'].notna().sum() > 0:
    # plot KDEs for both, clip extreme tails for clarity
    try:
        sns.kdeplot(df.loc[df['label_bin']==0, score_col].dropna(), label='Benign', bw_adjust=0.5)
        sns.kdeplot(df.loc[df['label_bin']==1, score_col].dropna(), label='Malicious', bw_adjust=0.5)
    except Exception:
        # fallback to hist
        plt.hist(df[score_col].dropna(), bins=100, alpha=0.7)
else:
    sns.kdeplot(df[score_col].dropna(), label='Scores', bw_adjust=0.5)

plt.title("Anomaly score distribution")
plt.xlabel(score_col)
plt.legend()
plt.tight_layout()
out = OUTDIR / "anomaly_score_distribution.png"
plt.savefig(out, dpi=150)
print("Saved:", out)
