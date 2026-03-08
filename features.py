import pandas as pd
import numpy as np

# Load flow-level data
flows_df = pd.read_csv('flows.csv')

# Handle missing or zero durations (avoid divide-by-zero)
flows_df['duration'] = flows_df['duration'].replace(0, np.nan)
flows_df['duration'] = flows_df['duration'].fillna(0.001)

# Add new derived features
flows_df['pkt_rate'] = flows_df['packet_count'] / flows_df['duration']
flows_df['byte_rate'] = flows_df['byte_count'] / flows_df['duration']
flows_df['size_iat_ratio'] = flows_df['avg_packet_size'] / (flows_df['mean_iat'] + 1e-6)  # avoid divide-by-zero

# Handle NaN/inf values
flows_df = flows_df.replace([np.inf, -np.inf], np.nan).fillna(0)

# Encode protocol (basic one-hot)
protocol_dummies = pd.get_dummies(flows_df['protocol'], prefix='proto')
features_df = pd.concat([flows_df, protocol_dummies], axis=1)

# Drop non-numeric / non-ML fields
features_df = features_df.drop(columns=[
    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'start_time', 'end_time', 'Source_File'
], errors='ignore')

# Save as features.csv
features_df.to_csv('features.csv', index=False)
print(f"Created 'features.csv' with {features_df.shape[1]} features and {len(features_df)} rows.")
