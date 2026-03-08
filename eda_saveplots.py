# eda_save_plots.py
"""
Batch EDA saving script for Phase 5.
Run: python eda_save_plots.py

Saves all outputs to eda_outputs/
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px

matplotlib.use('Agg')   # headless backend for saving
matplotlib.rcParams.update({'figure.max_open_warning': 20})

OUT_DIR = Path("eda_outputs")
OUT_DIR.mkdir(exist_ok=True)

# prefer combined_features
candidates = ["combined_features.csv", "combined_cic_ctu.csv", "combined_cic_ctu_parquet.parquet"]
df = None
for c in candidates:
    if Path(c).exists():
        print("Loading", c)
        if c.endswith(".parquet"):
            df = pd.read_parquet(c)
        else:
            df = pd.read_csv(c, low_memory=False)
        break

if df is None:
    print("No combined dataset found. Place combined_features.csv or combined_cic_ctu.csv in working dir.")
    sys.exit(1)

# ensure label bin
if 'label' in df.columns:
    df['label_bin'] = df['label'].astype(str).str.upper().apply(lambda x: 0 if 'BENIGN' in x or 'NORMAL' in x else 1)
elif 'label_bin' not in df.columns:
    df['label_bin'] = 0

# minimal safe conversions
def safe_num(s):
    return pd.to_numeric(df[s], errors='coerce') if s in df.columns else pd.Series(dtype=float)

# small helper to save fig
def save(fig, name):
    p = OUT_DIR / name
    fig.savefig(p, bbox_inches='tight', dpi=150)
    print("Saved", p)

# 1 Duration (log-binned)
if 'duration' in df.columns or 'dur' in df.columns:
    c = 'duration' if 'duration' in df.columns else 'dur'
    data_b = safe_num(c)[df['label_bin']==0].dropna()
    data_m = safe_num(c)[df['label_bin']==1].dropna()
    if len(data_b)>0 or len(data_m)>0:
        plt.figure(figsize=(8,4))
        a = max(1, data_b.max() if len(data_b)>0 else 1, data_m.max() if len(data_m)>0 else 1)
        bins = np.logspace(np.log10(1e-3), np.log10(a+1), 60)
        plt.hist(data_b, bins=bins, alpha=0.6, label='Benign')
        plt.hist(data_m, bins=bins, alpha=0.6, label='Malicious')
        plt.xscale('log')
        plt.xlabel('Duration (s) [log]')
        plt.ylabel('Count')
        plt.legend()
        plt.title("Duration: Benign vs Malicious")
        save(plt.gcf(), "duration_benign_vs_malicious.png")
        plt.close()
else:
    print("No duration column; skipping.")

# 2 Packet counts
pk_cols = None
for c in ['tot_pkts','totpkts','total_packets','total_fwd_packets']:
    if c in df.columns:
        pk_cols = c
        break
if pk_cols:
    data_b = pd.to_numeric(df[pk_cols][df['label_bin']==0], errors='coerce').dropna()
    data_m = pd.to_numeric(df[pk_cols][df['label_bin']==1], errors='coerce').dropna()
    plt.figure(figsize=(8,4))
    a = max(10, data_b.max() if len(data_b)>0 else 10, data_m.max() if len(data_m)>0 else 10)
    bins = np.logspace(0, np.log10(a+1), 40)
    plt.hist(data_b, bins=bins, alpha=0.6, label='Benign')
    plt.hist(data_m, bins=bins, alpha=0.6, label='Malicious')
    plt.xscale('log')
    plt.title("Packet count distribution")
    plt.xlabel("Total packets [log]")
    plt.legend()
    save(plt.gcf(), "pktcount_benign_vs_malicious.png")
    plt.close()
else:
    print("No packet count col found; skipping.")

# 3 Bytes
byte_cols = None
for c in ['tot_bytes','totbytes','total_bytes','fwd_bytes']:
    if c in df.columns:
        byte_cols = c
        break
if byte_cols:
    data_b = pd.to_numeric(df[byte_cols][df['label_bin']==0], errors='coerce').dropna()
    data_m = pd.to_numeric(df[byte_cols][df['label_bin']==1], errors='coerce').dropna()
    plt.figure(figsize=(8,4))
    a = max(10, data_b.max() if len(data_b)>0 else 10, data_m.max() if len(data_m)>0 else 10)
    bins = np.logspace(0, np.log10(a+1), 40)
    plt.hist(data_b, bins=bins, alpha=0.6, label='Benign')
    plt.hist(data_m, bins=bins, alpha=0.6, label='Malicious')
    plt.xscale('log')
    plt.title("Byte distribution")
    plt.legend()
    save(plt.gcf(), "bytes_benign_vs_malicious.png")
    plt.close()
else:
    print("No byte column; skipping.")

# 4 IAT
iat_c = next((c for c in ['flow_iat_mean','iat_mean','mean_iat'] if c in df.columns), None)
if iat_c:
    data_b = pd.to_numeric(df[iat_c][df['label_bin']==0], errors='coerce').dropna()
    data_m = pd.to_numeric(df[iat_c][df['label_bin']==1], errors='coerce').dropna()
    if len(data_b)>0 or len(data_m)>0:
        max_x = max(data_b.quantile(0.99) if len(data_b)>0 else 1, data_m.quantile(0.99) if len(data_m)>0 else 1)
        bins = np.linspace(0, max_x, 80)
        plt.figure(figsize=(8,4))
        plt.hist(data_b.clip(0,max_x), bins=bins, alpha=0.6, label='Benign')
        plt.hist(data_m.clip(0,max_x), bins=bins, alpha=0.6, label='Malicious')
        plt.title("IAT mean distribution (clipped 99%)")
        plt.legend()
        save(plt.gcf(), "iat_benign_vs_malicious.png")
        plt.close()
else:
    print("No IAT column; skipping.")

# 5 Protocols
proto_c = next((c for c in ['protocol','proto'] if c in df.columns), None)
if proto_c:
    proto = df[proto_c].fillna("UNKNOWN").astype(str).value_counts().nlargest(40)
    plt.figure(figsize=(9,4))
    proto.plot(kind='bar')
    plt.title("Top protocols")
    plt.ylabel("Count")
    save(plt.gcf(), "protocol_counts.png")
    plt.close()
else:
    print("No protocol column; skipping protocol counts.")

# 6 Correlation heatmap
candidates = []
for c in ['duration','fwd_pkts','bwd_pkts','tot_pkts','fwd_bytes','bwd_bytes','tot_bytes','flow_iat_mean','flow_iat_std']:
    if c in df.columns:
        candidates.append(c)
if candidates:
    nums = df[candidates].apply(pd.to_numeric, errors='coerce').fillna(0)
    corr = nums.corr()
    plt.figure(figsize=(10,8))
    im = plt.imshow(corr, cmap='RdYlBu', vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.03)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation heatmap")
    save(plt.gcf(), "correlation_heatmap.png")
    plt.close()
else:
    print("Not enough numeric columns for correlation heatmap.")

# 7 Anomaly score histogram + KDE (saved as PNG + interactive)
anom_c = next((c for c in ['anomaly_score','if_score','rerank_score','score'] if c in df.columns), None)
if anom_c:
    scores_b = pd.to_numeric(df[anom_c][df['label_bin']==0], errors='coerce').dropna()
    scores_m = pd.to_numeric(df[anom_c][df['label_bin']==1], errors='coerce').dropna()
    plt.figure(figsize=(8,4))
    plt.hist(scores_b, bins=100, alpha=0.6, label='Benign')
    plt.hist(scores_m, bins=100, alpha=0.6, label='Malicious')
    plt.title("Anomaly score distribution")
    plt.legend()
    save(plt.gcf(), "anomaly_score_distribution.png")
    plt.close()
    # interactive plotly
    hist_df = pd.DataFrame({anom_c: pd.concat([scores_b.rename('Benign'), scores_m.rename('Malicious')], axis=1).stack().reset_index(level=1,drop=True)})
    hist_df['label'] = ['Benign']*len(scores_b) + ['Malicious']*len(scores_m)
    fig = px.histogram(pd.concat([scores_b.rename('score').to_frame().assign(label='Benign'),
                                  scores_m.rename('score').to_frame().assign(label='Malicious')]).reset_index(),
                       x='score', color='label', nbins=120, title="Anomaly score distribution (interactive)")
    fig.write_html(str(OUT_DIR / "anomaly_score_distribution.html"))
    print("Saved interactive anomaly score HTML.")
else:
    print("No anomaly score column found; skipping.")

# 8 Timeline (interactive) if time-like column exists
time_c = next((c for c in ['start_time','ts','time','timestamp'] if c in df.columns), None)
if time_c:
    try:
        df['__ts'] = pd.to_datetime(df[time_c], errors='coerce')
        df_ts = df.dropna(subset=['__ts'])
        agg = df_ts.set_index('__ts').resample('1Min').agg({candidates[0]: 'sum' if candidates else 'size', 'label_bin': 'sum'})
        agg = agg.fillna(0)
        fig = px.line(agg.reset_index(), x='__ts', y=agg.columns[0], title="Traffic volume over time (per minute)")
        fig.write_html(str(OUT_DIR / "timeline_traffic.html"))
        print("Saved timeline interactive:", OUT_DIR / "timeline_traffic.html")
    except Exception as e:
        print("Timeline creation failed:", e)
else:
    print("No time-like column; skipping timeline.")
