import pyshark
import csv

cap = pyshark.FileCapture('https1.pcapng', keep_packets=False)

output_file = 'detailed_packets.csv'
fieldnames = [
    'No', 'Time', 'Source_MAC', 'Destination_MAC',
    'Source_IP', 'Destination_IP',
    'Source_Port', 'Destination_Port',
    'Protocol', 'Length', 'Info'
]

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i, packet in enumerate(cap, start=1):
        try:
            data = {key: 'N/A' for key in fieldnames}
            data['No'] = i
            data['Time'] = packet.sniff_time
            data['Length'] = packet.length
            data['Protocol'] = packet.highest_layer

            # MAC addresses
            if hasattr(packet, 'eth'):
                data['Source_MAC'] = packet.eth.src
                data['Destination_MAC'] = packet.eth.dst

            # IP addresses
            if hasattr(packet, 'ip'):
                data['Source_IP'] = packet.ip.src
                data['Destination_IP'] = packet.ip.dst

            # Ports (TCP/UDP)
            if hasattr(packet, 'tcp'):
                data['Source_Port'] = packet.tcp.srcport
                data['Destination_Port'] = packet.tcp.dstport
            elif hasattr(packet, 'udp'):
                data['Source_Port'] = packet.udp.srcport
                data['Destination_Port'] = packet.udp.dstport

            # Info
            data['Info'] = str(packet)

            writer.writerow(data)

        except Exception as e:
            print(f"Skipping packet {i}: {e}")
            continue

