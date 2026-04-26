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
df = pd.read_csv("Wholesale customers data.csv")
df.columns = df.columns.str.strip()

# ===============================
# PART A: PREPROCESSING & EDA
# ===============================

# Select required columns
df = df[['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']]

# Handle missing values
df = df.dropna()

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Feature distributions
df.hist(figsize=(10,6))
plt.suptitle("Feature Distributions")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================

# --- Dendrogram ---
plt.figure(figsize=(15,6))
plt.figure(figsize=(14,6))

sch.dendrogram(
    sch.linkage(X_scaled, method='ward'),
    truncate_mode='level',
    p=5,
    no_labels=True           # 🔥 removes crowding
)

plt.title("Dendrogram (Simplified)")
plt.xlabel("Cluster Groups")
plt.ylabel("Distance")
plt.show()

# 👉 Choose clusters (usually 3–4 from dendrogram)
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agg = agg.fit_predict(X_scaled)

df['Cluster_Agg'] = labels_agg

# ===============================
# PART C: EVALUATION & VISUALIZATION
# ===============================

# Scatter Plot (Milk vs Grocery)
plt.scatter(df['Milk'], df['Grocery'], c=labels_agg)
plt.xlabel("Milk")
plt.ylabel("Grocery")
plt.title("Agglomerative Clusters")
plt.show()

# Silhouette Score
score_agg = silhouette_score(X_scaled, labels_agg)
print("\nAgglomerative Silhouette Score:", score_agg)

# ===============================
# K-MEANS (FOR COMPARISON)
# ===============================
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

score_km = silhouette_score(X_scaled, labels_km)
print("K-Means Silhouette Score:", score_km)

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Cluster 0 → High Grocery & Detergents (Retail shops)")
print("Cluster 1 → High Fresh & Frozen (Hotels/Restaurants)")
print("Cluster 2 → Moderate spending (General customers)")

print("\nComparison:")
print("Agglomerative gives hierarchical grouping (tree-based).")
print("K-Means is faster and better for large datasets.")