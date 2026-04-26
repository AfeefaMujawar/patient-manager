# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

# ===============================
# SYNTHETIC DATASET (10 rows)
# ===============================
df = pd.DataFrame({
    'Alcohol':[13.2,12.8,13.5,12.9,14.2,13.8,12.5,13.1,14.0,13.6],
    'MalicAcid':[2.3,1.8,2.6,2.0,3.1,2.9,1.7,2.2,3.0,2.7],
    'Ash':[2.4,2.1,2.5,2.2,2.8,2.6,2.0,2.3,2.7,2.5],
    'Magnesium':[100,95,110,98,120,115,90,105,118,112]
})

# ===============================
# PART A: NORMALIZATION + EDA
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Feature distributions
df.hist(figsize=(8,5))
plt.suptitle("Feature Distributions")
plt.show()

# Feature relationship
plt.scatter(df['Alcohol'], df['MalicAcid'])
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.title("Feature Relationship")
plt.show()

# ===============================
# PART B: AGGLOMERATIVE CLUSTERING
# ===============================

# Dendrogram (clean view)
plt.figure(figsize=(12,5))
sch.dendrogram(
    sch.linkage(X_scaled, method='ward'),
    truncate_mode='lastp',
    p=10
)
plt.title("Dendrogram")
plt.xlabel("Clusters")
plt.ylabel("Distance")
plt.show()

# Optimal clusters = 3
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agg = agg.fit_predict(X_scaled)

df['Cluster_Agg'] = labels_agg

# ===============================
# K-MEANS (FOR COMPARISON)
# ===============================
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

df['Cluster_KMeans'] = labels_km

# ===============================
# PART C: EVALUATION
# ===============================

# Visualization (Agglomerative)
plt.scatter(df['Alcohol'], df['MalicAcid'], c=labels_agg)
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.title("Agglomerative Clusters")
plt.show()

# Visualization (K-Means)
plt.scatter(df['Alcohol'], df['MalicAcid'], c=labels_km)
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.title("K-Means Clusters")
plt.show()

# Silhouette Scores
score_agg = silhouette_score(X_scaled, labels_agg)
score_km = silhouette_score(X_scaled, labels_km)

print("\nAgglomerative Silhouette Score:", score_agg)
print("K-Means Silhouette Score:", score_km)

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Cluster 0 → Low alcohol wines (light)")
print("Cluster 1 → Moderate wines")
print("Cluster 2 → High alcohol & magnesium wines (premium)")

print("\nComparison:")
print("Agglomerative shows hierarchical relationships (tree-based).")
print("K-Means gives compact spherical clusters and is faster.")