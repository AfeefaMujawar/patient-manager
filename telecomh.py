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
    'Tenure': [1, 3, 6, 12, 24, 2, 8, 15, 30, 36],
    'MonthlyCharges': [20, 30, 50, 60, 80, 25, 55, 65, 90, 100],
    'TotalCharges': [20, 90, 300, 720, 1920, 50, 440, 975, 2700, 3600]
}
df = pd.DataFrame(data)

# ===============================
# PART A: CLEANING + EDA
# ===============================
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary:\n", df.describe())

# Distribution
df.hist()
plt.show()

# Relationship
plt.scatter(df['Tenure'], df['TotalCharges'])
plt.xlabel("Tenure")
plt.ylabel("Total Charges")
plt.title("Tenure vs Total Charges")
plt.show()

# ===============================
# NORMALIZATION
# ===============================
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
# Hierarchical
plt.scatter(df['Tenure'], df['TotalCharges'], c=labels_hc)
plt.xlabel("Tenure")
plt.ylabel("Total Charges")
plt.title("Hierarchical Clusters")
plt.show()

# K-Means
plt.scatter(df['Tenure'], df['TotalCharges'], c=labels_km)
plt.xlabel("Tenure")
plt.ylabel("Total Charges")
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
print("Cluster 0 → New customers (low tenure, low charges)")
print("Cluster 1 → Mid-term customers")
print("Cluster 2 → Long-term high-value customers")

print("\nComparison:")
print("Hierarchical → Shows structure via dendrogram")
print("K-Means → Faster and forms compact clusters")

print("\nConclusion:")
print("Both methods identify similar telecom customer segments.")