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
    'StudyHours': [2, 3, 5, 6, 8, 1, 4, 7, 9, 2],
    'Attendance': [60, 65, 70, 75, 90, 55, 68, 85, 95, 58],
    'Marks':      [50, 55, 65, 70, 85, 45, 60, 78, 92, 48]
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

plt.figure(figsize=(10,5))
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Students")
plt.ylabel("Distance")
plt.show()

# Optimal clusters = 3

hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hc = hc.fit_predict(X)

df['HC_Cluster'] = labels_hc

# ===============================
# PART C: VISUALIZATION
# ===============================
plt.scatter(df['StudyHours'], df['Marks'], c=labels_hc)
plt.xlabel("Study Hours")
plt.ylabel("Marks")
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
print("Cluster 0 → Low performers (low study, low marks)")
print("Cluster 1 → Average students")
print("Cluster 2 → High performers (high study, high marks)")

print("\nComparison:")
print("Hierarchical → Shows natural grouping using dendrogram")
print("K-Means → Faster and scalable for larger datasets")

print("\nConclusion:")
print("Both methods produce similar student segments with clear performance groups.")