# data_checks.py
import pandas as pd
from pathlib import Path

p = Path("combined_cic_ctu.csv")
if not p.exists():
    raise FileNotFoundError("combined_cic_ctu.csv not found")

df = pd.read_csv(p, low_memory=False)
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# Show distinct values & counts for label & protocol
print("\nLabel unique values (sample):", df['label'].astype(str).unique()[:30])
if 'protocol' in df.columns:
    print("\nProtocol unique value counts (top 50):")
    print(df['protocol'].value_counts().head(50))
else:
    print("\nNo 'protocol' column present.")

# Show a few rows where label contains 'BENIGN' exactly (case-insensitive)
print("\nSample rows where label == 'BENIGN' (case-insensitive):")
print(df[df['label'].astype(str).str.strip().str.upper()=="BENIGN"].head(5))

# Count strict benign vs non-benign
strict_benign = (df['label'].astype(str).str.strip().str.upper()=="BENIGN").sum()
total = len(df)
print(f"\nStrict BENIGN: {strict_benign:,} of {total:,} rows ({strict_benign/total:.4%})")
