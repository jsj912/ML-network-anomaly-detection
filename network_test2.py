# generate_network_flows.py
import csv
import math
import random
import numpy as np
import pandas as pd

PROTO_CHOICES = ['tcp', 'udp', 'icmp']

def make_benign(n):
    rows = []
    for _ in range(n):
        duration = round(random.expovariate(1/20) + 0.1, 3)
        fwd_pkts = max(1, int(np.random.poisson(6)))
        bwd_pkts = max(0, int(np.random.poisson(5)))
        fwd_bytes = fwd_pkts * random.randint(60, 1200)
        bwd_bytes = bwd_pkts * random.randint(40, 900)
        iat_mean = round(random.uniform(0.01, 1.5), 4)
        iat_std = round(iat_mean * random.uniform(0.01, 0.5), 4)
        proto = random.choice(['tcp','udp'])
        rows.append((duration,fwd_pkts,bwd_pkts,fwd_bytes,bwd_bytes,iat_mean,iat_std,proto,'benign'))
    return rows

def make_mirai(n):
    # Mirai-like: short duration, huge UDP floods, many pkts, low iat
    rows = []
    for _ in range(n):
        duration = round(random.uniform(0.1, 10), 3)
        fwd_pkts = random.randint(40, 500)
        bwd_pkts = random.randint(0, 2)
        fwd_bytes = fwd_pkts * random.randint(60, 1500)
        bwd_bytes = 0
        iat_mean = round(random.uniform(0.0001, 0.02), 6)
        iat_std = round(iat_mean * random.uniform(0.01, 0.3), 6)
        rows.append((duration,fwd_pkts,bwd_pkts,fwd_bytes,bwd_bytes,iat_mean,iat_std,'udp','mirai'))
    return rows

def make_rat_beacon(n):
    # RAT beacon: periodic, small packets, relatively uniform iat
    rows = []
    for _ in range(n):
        duration = round(random.uniform(0.5, 30), 3)
        fwd_pkts = random.randint(1, 6)
        bwd_pkts = random.randint(0, 4)
        fwd_bytes = fwd_pkts * random.randint(40, 300)
        bwd_bytes = bwd_pkts * random.randint(40, 300)
        iat_mean = round(random.uniform(0.5, 5.0), 4)
        iat_std = round(iat_mean * random.uniform(0.01, 0.2), 4)
        proto = random.choice(['tcp','udp'])
        rows.append((duration,fwd_pkts,bwd_pkts,fwd_bytes,bwd_bytes,iat_mean,iat_std,proto,'rat_beacon'))
    return rows

def make_ransomware_c2(n):
    # Ransomware C2: occasional small flows + some larger command flows; asymmetry possible
    rows = []
    for _ in range(n):
        if random.random() < 0.7:
            # small C2 beacon
            duration = round(random.uniform(0.1, 5), 3)
            fwd_pkts = random.randint(1, 6)
            bwd_pkts = random.randint(0, 4)
            fwd_bytes = fwd_pkts * random.randint(40, 300)
            bwd_bytes = bwd_pkts * random.randint(40, 800)
        else:
            # command / exfil flow
            duration = round(random.uniform(1, 60), 3)
            fwd_pkts = random.randint(10, 200)
            bwd_pkts = random.randint(0, 30)
            fwd_bytes = fwd_pkts * random.randint(200, 1500)
            bwd_bytes = bwd_pkts * random.randint(40, 2000)
        iat_mean = round(random.uniform(0.01, 1.0), 4)
        iat_std = round(iat_mean * random.uniform(0.01, 0.6), 4)
        proto = 'tcp'
        rows.append((duration,fwd_pkts,bwd_pkts,fwd_bytes,bwd_bytes,iat_mean,iat_std,proto,'ransomware_c2'))
    return rows

def make_dns_tunnel(n):
    # DNS tunneling: small packets but many short flows with tiny payloads, maybe many queries (UDP)
    rows = []
    for _ in range(n):
        duration = round(random.uniform(0.01, 2.0), 4)
        fwd_pkts = random.randint(5, 60)
        bwd_pkts = random.randint(0, 5)
        fwd_bytes = fwd_pkts * random.randint(50, 150)
        bwd_bytes = bwd_pkts * random.randint(40, 120)
        iat_mean = round(random.uniform(0.001, 0.1), 5)
        iat_std = round(iat_mean * random.uniform(0.01, 0.4), 5)
        rows.append((duration,fwd_pkts,bwd_pkts,fwd_bytes,bwd_bytes,iat_mean,iat_std,'udp','dns_tunnel'))
    return rows

def make_portscan_like(n):
    rows = []
    for _ in range(n):
        duration = round(random.uniform(0.01, 10), 4)
        fwd_pkts = random.randint(20, 500)
        bwd_pkts = random.randint(0, 1)
        fwd_bytes = fwd_pkts * random.randint(40, 200)
        bwd_bytes = 0
        iat_mean = round(random.uniform(0.0001, 0.05), 6)
        iat_std = round(iat_mean * random.uniform(0.01, 0.2), 6)
        proto = random.choice(['tcp','icmp','udp'])
        rows.append((duration,fwd_pkts,bwd_pkts,fwd_bytes,bwd_bytes,iat_mean,iat_std,proto,'portscan'))
    return rows

def generate_mixed(out_csv='mixed_dataset_5000.csv', n_total=5000, seed=42):
    random.seed(seed)
    parts = []
    parts += make_benign(int(n_total*0.6))
    parts += make_mirai(int(n_total*0.08))
    parts += make_rat_beacon(int(n_total*0.06))
    parts += make_ransomware_c2(int(n_total*0.06))
    parts += make_dns_tunnel(int(n_total*0.06))
    parts += make_portscan_like(int(n_total*0.04))
    # shuffle and save
    random.shuffle(parts)
    df = pd.DataFrame(parts, columns=['duration','fwd_pkts','bwd_pkts','fwd_bytes','bwd_bytes','flow_iat_mean','flow_iat_std','protocol','label'])
    df.to_csv(out_csv, index=False)
    print("Saved", out_csv, "with", len(df), "rows")
    return df

if __name__ == "__main__":
    # small example files
    df_small = pd.DataFrame(make_benign(10) + make_portscan_like(5) + make_mirai(5),
                            columns=['duration','fwd_pkts','bwd_pkts','fwd_bytes','bwd_bytes','flow_iat_mean','flow_iat_std','protocol','label'])
    df_small.to_csv('test_flows_small_labeled.csv', index=False)
    # big mixed dataset
    generate_mixed(out_csv='mixed_dataset_5000.csv', n_total=5000)
