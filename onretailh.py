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
# LOAD DATASET
# ===============================
df = pd.read_csv("OnlineRetail.csv", encoding='latin1')
df.columns = df.columns.str.strip()

# ===============================
# PART A: CLEANING + AGGREGATION
# ===============================

# Remove missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Remove invalid entries
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Convert CustomerID
df['CustomerID'] = df['CustomerID'].astype(int)

# Aggregate per customer
customer_df = df.groupby('CustomerID').agg({
    'Quantity':'sum',
    'UnitPrice':'mean'
}).reset_index()

# Scaling
X = customer_df[['Quantity','UnitPrice']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# PART B: AGGLOMERATIVE CLUSTERING
# ===============================

# Dendrogram (clean)
plt.figure(figsize=(12,5))
Z = sch.linkage(X_scaled, method='ward')
sch.dendrogram(Z, truncate_mode='lastp', p=20)
plt.title("Dendrogram")
plt.show()

# Choose clusters (3)
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agg = agg.fit_predict(X_scaled)

customer_df['Cluster_Agg'] = labels_agg

# ===============================
# K-MEANS (FOR COMPARISON)
# ===============================
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

customer_df['Cluster_KMeans'] = labels_km

# ===============================
# PART C: VISUALIZATION
# ===============================

# Agglomerative
plt.scatter(customer_df['Quantity'], customer_df['UnitPrice'], c=labels_agg)
plt.xlabel("Quantity")
plt.ylabel("Unit Price")
plt.title("Agglomerative Clusters")
plt.show()

# K-Means
plt.scatter(customer_df['Quantity'], customer_df['UnitPrice'], c=labels_km)
plt.xlabel("Quantity")
plt.ylabel("Unit Price")
plt.title("K-Means Clusters")
plt.show()

# ===============================
# EVALUATION
# ===============================
score_agg = silhouette_score(X_scaled, labels_agg)
score_km = silhouette_score(X_scaled, labels_km)

print("\nSilhouette Scores:")
print("Agglomerative:", score_agg)
print("K-Means:", score_km)

# Cluster size comparison
print("\nCluster Sizes (Agglomerative):\n", customer_df['Cluster_Agg'].value_counts())
print("\nCluster Sizes (K-Means):\n", customer_df['Cluster_KMeans'].value_counts())

# Cross-tab comparison
print("\nCluster Agreement:\n", pd.crosstab(customer_df['Cluster_Agg'], customer_df['Cluster_KMeans']))

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Cluster 0 → Low purchase customers")
print("Cluster 1 → Bulk buyers (high quantity)")
print("Cluster 2 → Medium spending customers")

print("\nConclusion:")
if score_km > score_agg:
    print("K-Means gives better compact clusters.")
else:
    print("Agglomerative provides better hierarchical grouping.")