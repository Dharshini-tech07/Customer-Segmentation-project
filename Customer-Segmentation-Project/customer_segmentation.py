import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_excel("Sample_Superstore.xls")

# Select features
data = df[['Sales', 'Profit']]

# Create KMeans model
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(data)

# Plot clusters
plt.figure(figsize=(8,6))
plt.scatter(df['Sales'], df['Profit'], c=df['Cluster'], cmap='viridis')

# Cluster centers
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    color='red',
    s=200,
    marker='X'
)

plt.title("Customer Segmentation")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.show()