import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('detailed_packets.csv')

# Handle missing values
df.fillna('N/A', inplace=True)

protocol_counts = df['Protocol'].value_counts()

plt.figure(figsize=(8, 6))
protocol_counts.plot(kind='bar', color='skyblue')
plt.title('Packet Distribution by Protocol')
plt.xlabel('Protocol')
plt.ylabel('Number of Packets')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

src_counts = df['Source_IP'].value_counts().head(10)

plt.figure(figsize=(8, 6))
src_counts.plot(kind='bar', color='lightgreen')
plt.title('Top 10 Source IP Addresses')
plt.xlabel('Source IP')
plt.ylabel('Number of Packets')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

dst_counts = df['Destination_IP'].value_counts().head(10)

plt.figure(figsize=(8, 6))
dst_counts.plot(kind='bar', color='salmon')
plt.title('Top 10 Destination IP Addresses')
plt.xlabel('Destination IP')
plt.ylabel('Number of Packets')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
