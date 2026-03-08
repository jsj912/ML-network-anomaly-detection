import pandas as pd

files = [
    'https1_packets.csv',
    'wifitraffic1_packets.csv',
    'wifitraffic2_packets.csv'
]

dfs = []
for file in files:
    df = pd.read_csv(file)
    df['Source_File'] = file 
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)

merged_df.to_csv('all_packets.csv', index=False)

