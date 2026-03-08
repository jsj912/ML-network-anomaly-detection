# alternative streaming approach (append to parquet in chunks)
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CSV = "two_stage_scores.csv"
PARQUET = "two_stage_scores.parquet"
first = True
for chunk in pd.read_csv(CSV, chunksize=250_000, engine='python', on_bad_lines='skip'):
    table = pa.Table.from_pandas(chunk)
    if first:
        pq.write_table(table, PARQUET)
        first = False
    else:
        with pq.ParquetWriter(PARQUET, table.schema, use_dictionary=True) as writer:
            writer.write_table(table)
