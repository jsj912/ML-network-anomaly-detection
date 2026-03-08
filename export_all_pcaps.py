import pyshark
import csv
import sys
from pathlib import Path

pcap_files = ['https1.pcapng', 'wifitraffic1.pcapng', 'wifitraffic2.pcapng'] 
fieldnames = [
    'No', 'Time', 'Source_MAC', 'Destination_MAC',
    'Source_IP', 'Destination_IP',
    'Source_Port', 'Destination_Port',
    'Protocol', 'Length', 'Info'
]

def export_one(pcap_path, out_csv):
    print(f"Exporting {pcap_path} -> {out_csv}")
    cap = pyshark.FileCapture(str(pcap_path), keep_packets=False)
    with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, packet in enumerate(cap, start=1):
            try:
                data = {key: 'N/A' for key in fieldnames}
                data['No'] = i
                data['Time'] = packet.sniff_time
                data['Length'] = packet.length
                data['Protocol'] = packet.highest_layer

                if hasattr(packet, 'eth'):
                    data['Source_MAC'] = packet.eth.src
                    data['Destination_MAC'] = packet.eth.dst

                if hasattr(packet, 'ip'):
                    data['Source_IP'] = packet.ip.src
                    data['Destination_IP'] = packet.ip.dst

                if hasattr(packet, 'tcp'):
                    data['Source_Port'] = packet.tcp.srcport
                    data['Destination_Port'] = packet.tcp.dstport
                elif hasattr(packet, 'udp'):
                    data['Source_Port'] = packet.udp.srcport
                    data['Destination_Port'] = packet.udp.dstport

                data['Info'] = str(packet)
                writer.writerow(data)
            except Exception as e:
                print(f"Skipping packet {i}: {e}")
                continue
    cap.close()
    print(f"Saved {out_csv}")

if __name__ == "__main__":
    for p in pcap_files:
        p_path = Path(p)
        if not p_path.exists():
            print(f"File not found: {p}; skipping.")
            continue
        out_name = p_path.stem + "_packets.csv"
        export_one(p_path, out_name)
