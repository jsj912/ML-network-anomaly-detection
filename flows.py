import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('all_packets.csv')

df = df.dropna(subset=['Source_IP', 'Destination_IP', 'Protocol', 'Length', 'Time'])
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
df['Source_Port'] = pd.to_numeric(df['Source_Port'], errors='coerce').fillna(0)
df['Destination_Port'] = pd.to_numeric(df['Destination_Port'], errors='coerce').fillna(0)

df = df.sort_values('Time')

flow_keys = ['Source_IP', 'Destination_IP', 'Source_Port', 'Destination_Port', 'Protocol']

flows = []
for flow_id, group in df.groupby(flow_keys):
    group = group.sort_values('Time')
    pkt_times = group['Time'].astype('int64') // 1_000_000_000  # nanoseconds to seconds
    iats = np.diff(pkt_times)

    flow_data = {
        'src_ip': flow_id[0],
        'dst_ip': flow_id[1],
        'src_port': flow_id[2],
        'dst_port': flow_id[3],
        'protocol': flow_id[4],
        'packet_count': len(group),
        'byte_count': group['Length'].sum(),
        'start_time': group['Time'].min(),
        'end_time': group['Time'].max(),
        'duration': (group['Time'].max() - group['Time'].min()).total_seconds(),
        'avg_packet_size': group['Length'].mean(),
        'std_packet_size': group['Length'].std(),
        'mean_iat': np.mean(iats) if len(iats) > 0 else 0,
        'std_iat': np.std(iats) if len(iats) > 0 else 0,
        'Source_File': group['Source_File'].iloc[0],
    }
    flows.append(flow_data)

flows_df = pd.DataFrame(flows)

flows_df.to_csv('flows.csv', index=False)   
