# check_scores_locations.py
import os, pandas as pd
candidates = ["combined_features.csv", "combined_cic_ctu.csv", "two_stage_scores.csv",
              "if_anomaly_scores.csv", "anomaly_scores_features.csv", "if_scores_per_protocol.csv"]

for f in candidates:
    if not os.path.exists(f):
        print(f"{f}: NOT FOUND")
        continue
    print(f"\n{f}:")
    try:
        df = pd.read_csv(f, nrows=5)
        print(" cols:", list(df.columns))
        # look for score-like names
        for name in ['anomaly_score','if_score','score','rerank_score','if_score','if_decision','if_score_raw','anomaly']:
            if name in df.columns:
                print("  -> Found score column:", name)
    except Exception as e:
        print("  failed to read:", e)
