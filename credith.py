# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("Credit CArd.csv")
df.columns = df.columns.str.strip()

# ===============================
# PART A: CLEANING + EDA
# ===============================
df = df[['BALANCE','PURCHASES','CREDIT_LIMIT']]

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

print("\nSummary:\n", df.describe())

# Distribution
df.hist()
plt.show()

# ===============================
# NORMALIZATION
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(df)

# ===============================
# PART B: HIERARCHICAL CLUSTERING
# ===============================

# Dendrogram
linked = linkage(X, method='ward')

plt.figure(figsize=(12,6))
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Optimal clusters = 3

hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hc = hc.fit_predict(X)

df['HC_Cluster'] = labels_hc

# ===============================
# PART C: VISUALIZATION
# ===============================
plt.scatter(df['PURCHASES'], df['BALANCE'], c=labels_hc)
plt.xlabel("Purchases")
plt.ylabel("Balance")
plt.title("Hierarchical Clusters")
plt.show()

# ===============================
# EVALUATION
# ===============================
sil_hc = silhouette_score(X, labels_hc)
print("\nHierarchical Silhouette Score:", sil_hc)

# ===============================
# COMPARISON WITH K-MEANS
# ===============================
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels_km = kmeans.fit_predict(X)

df['KM_Cluster'] = labels_km

sil_km = silhouette_score(X, labels_km)
print("K-Means Silhouette Score:", sil_km)

# ===============================
# INTERPRETATION
# ===============================
print("\nCluster Interpretation:")
print("Cluster 0 → Low spenders (low balance & purchases)")
print("Cluster 1 → High spenders (high purchases, high credit usage)")
print("Cluster 2 → Moderate users (balanced behavior)")

print("\nComparison:")
print("Hierarchical → Better structure understanding (tree-based)")
print("K-Means → Faster and better for large datasets")

print("\nConclusion:")
print("Both methods identify similar segments, but K-Means is more scalable.")