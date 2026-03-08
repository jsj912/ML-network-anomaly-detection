import pyshark
import pandas as pd

cap = pyshark.FileCapture(r'https1.pcapng')

packets = []
for pkt in cap:
    try:
        packet_info = {
            'timestamp': float(pkt.sniff_timestamp),
            'src_ip': pkt.ip.src,
            'dst_ip': pkt.ip.dst,
            'protocol': pkt.transport_layer,
            'length': int(pkt.length)
        }
        if hasattr(pkt, pkt.transport_layer.lower()):
            layer = getattr(pkt, pkt.transport_layer.lower())
            if hasattr(layer, 'srcport'):
                packet_info['src_port'] = layer.srcport
            if hasattr(layer, 'dstport'):
                packet_info['dst_port'] = layer.dstport
        else:
            packet_info['src_port'] = None
            packet_info['dst_port'] = None

        packets.append(packet_info)
    except AttributeError:
        continue

cap.close()

df = pd.DataFrame(packets)
df.to_csv('packet_level.csv', index=False)
print(df.head())
