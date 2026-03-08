# merge_scores_chunked.py
import os
import pandas as pd
from pathlib import Path

FEATURES = "combined_cic_ctu.csv"   # your features file
SCORES_IF = "if_anomaly_scores.csv"
SCORES_TWO = "two_stage_scores.csv"
OUT = "combined_with_scores.csv"
CHUNKSIZE = 200_000  # adjust to memory (lower if necessary)

def choose_score_source():
    if Path(SCORES_IF).exists():
        print("Using", SCORES_IF, "-> 'anomaly_score'")
        return SCORES_IF, "anomaly_score"
    if Path(SCORES_TWO).exists():
        # read header to find best column
        hdr = pd.read_csv(SCORES_TWO, nrows=1)
        for pref in ("if_score","anomaly_score","rerank_score","score"):
            for c in hdr.columns:
                if pref == c or pref in c.lower():
                    print("Using", SCORES_TWO, "->", c)
                    return SCORES_TWO, c
        # fallback
        print("Using", SCORES_TWO, "-> fallback first numeric score column")
        for c in hdr.columns:
            if hdr[c].dtype.kind in 'fi':
                return SCORES_TWO, c
    raise SystemExit("No suitable scores file found. Place if_anomaly_scores.csv or two_stage_scores.csv beside this script.")

def chunked_merge(features_path, scores_path, score_col, out_path, chunksize=200000):
    print("Merging in chunks:", features_path, "<--", scores_path, "(", score_col, ")")
    feat_reader = pd.read_csv(features_path, chunksize=chunksize, low_memory=False)
    score_reader = pd.read_csv(scores_path, chunksize=chunksize, low_memory=False)

    first = True
    written = 0
    for fchunk, schunk in zip(feat_reader, score_reader):
        # align by index position: reset indexes and concat horizontally
        fchunk = fchunk.reset_index(drop=True)
        schunk = schunk.reset_index(drop=True)
        if score_col not in schunk.columns:
            # try lower-case match
            candidates = [c for c in schunk.columns if score_col.lower() in c.lower()]
            if candidates:
                col = candidates[0]
            else:
                raise ValueError(f"Score column {score_col} not found in chunk columns: {schunk.columns}")
        else:
            col = score_col
        # take only score col (and possibly label)
        keep_cols = [col]
        if 'label' in schunk.columns:
            keep_cols.append('label')
        keep = schunk[keep_cols].copy()
        # rename score column to a consistent name
        keep = keep.rename(columns={col: 'anomaly_score' if 'anomaly' in col.lower() or 'if' in col.lower() else col})
        out_chunk = pd.concat([fchunk.reset_index(drop=True), keep.reset_index(drop=True)], axis=1)

        if first:
            out_chunk.to_csv(out_path, index=False, mode='w')
            first = False
        else:
            out_chunk.to_csv(out_path, index=False, header=False, mode='a')
        written += len(out_chunk)
        print("Wrote rows:", written)
    print("Done. Output:", out_path)

if __name__ == "__main__":
    scores_file, score_column = choose_score_source()
    if not Path(FEATURES).exists():
        raise SystemExit(f"Features file not found: {FEATURES}")
    chunked_merge(FEATURES, scores_file, score_column, OUT, CHUNKSIZE)
