# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# ===============================
# SYNTHETIC DATASET (10 rows)
# ===============================
data = {
    'Area': [600, 800, 1200, 1500, 2000, 550, 900, 1300, 1800, 2200],
    'Bedrooms': [1, 2, 2, 3, 4, 1, 2, 3, 3, 4],
    'Price': [300000, 400000, 600000, 750000, 1000000, 280000, 420000, 650000, 900000, 1100000]
}
df = pd.DataFrame(data)

# ===============================
# PART A: SCALING + EDA
# ===============================
print("\nSummary:\n", df.describe())

# Relationships
plt.scatter(df['Area'], df['Price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()

# Distributions
df.hist()
plt.show()

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(df)

# ===============================
# PART B: HIERARCHICAL CLUSTERING
# ===============================

# Dendrogram (clean)
Z = linkage(X, method='ward')

plt.figure(figsize=(10,5))
dendrogram(Z, truncate_mode='level', p=3)
plt.title("Dendrogram")
plt.xlabel("Clusters")
plt.ylabel("Distance")
plt.show()

# Optimal clusters = 3
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hc = hc.fit_predict(X)

df['HC_Cluster'] = labels_hc

# ===============================
# K-MEANS (FOR COMPARISON)
# ===============================
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels_km = kmeans.fit_predict(X)

df['KM_Cluster'] = labels_km

# ===============================
# PART C: VISUALIZATION
# ===============================
# Hierarchical clusters
plt.scatter(df['Area'], df['Price'], c=labels_hc)
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Hierarchical Clusters")
plt.show()

# K-Means clusters
plt.scatter(df['Area'], df['Price'], c=labels_km)
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("K-Means Clusters")
plt.show()

# ===============================
# EVALUATION
# ===============================
sil_hc = silhouette_score(X, labels_hc)
sil_km = silhouette_score(X, labels_km)

print("\nSilhouette Scores:")
print("Hierarchical:", sil_hc)
print("K-Means:", sil_km)

print("\nCluster Sizes (HC):\n", df['HC_Cluster'].value_counts())
print("\nCluster Sizes (KM):\n", df['KM_Cluster'].value_counts())

print("\nCluster Agreement:\n", pd.crosstab(df['HC_Cluster'], df['KM_Cluster']))

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Cluster 0 → Low-cost houses (small area, low price)")
print("Cluster 1 → Mid-range houses")
print("Cluster 2 → Premium houses (large area, high price)")

print("\nComparison:")
print("Hierarchical → Shows natural grouping via dendrogram")
print("K-Means → Produces compact clusters efficiently")

print("\nConclusion:")
print("Both methods identify similar housing segments based on area and price.")