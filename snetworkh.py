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
df = pd.read_csv("Social_Network_Ads.csv")
df.columns = df.columns.str.strip()
df = df[['Age','EstimatedSalary']]

# ===============================
# PART A: SCALING + EDA
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

df.hist(figsize=(8,4))
plt.suptitle("Feature Distributions")
plt.show()

plt.scatter(df['Age'], df['EstimatedSalary'])
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("User Distribution")
plt.show()

# ===============================
# PART B: HIERARCHICAL
# ===============================
plt.figure(figsize=(12,5))
Z = sch.linkage(X_scaled, method='ward')
sch.dendrogram(Z, truncate_mode='lastp', p=20)
plt.title("Dendrogram")
plt.show()

agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agg = agg.fit_predict(X_scaled)
df['Cluster_Agg'] = labels_agg

# ===============================
# K-MEANS
# ===============================
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)
df['Cluster_KMeans'] = labels_km

# ===============================
# PART C: VISUALIZATION
# ===============================
plt.scatter(df['Age'], df['EstimatedSalary'], c=labels_agg)
plt.title("Agglomerative Clusters")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()

plt.scatter(df['Age'], df['EstimatedSalary'], c=labels_km)
plt.title("K-Means Clusters")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()

# ===============================
# EVALUATION
# ===============================
score_agg = silhouette_score(X_scaled, labels_agg)
score_km = silhouette_score(X_scaled, labels_km)

print("\nSilhouette Scores:")
print("Agglomerative:", score_agg)
print("K-Means:", score_km)

# ===============================
# 🔥 COMPARISON (IMPORTANT)
# ===============================
print("\nCluster Count Comparison:")
print("Agglomerative:\n", df['Cluster_Agg'].value_counts())
print("\nK-Means:\n", df['Cluster_KMeans'].value_counts())

print("\nCluster Agreement (Cross-tab):")
print(pd.crosstab(df['Cluster_Agg'], df['Cluster_KMeans']))

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Both methods segment users into low, medium, high income groups.")
print("K-Means forms tighter clusters, Agglomerative shows hierarchy.")

print("\nConclusion:")
if score_km > score_agg:
    print("K-Means performs better for this dataset.")
else:
    print("Agglomerative performs better for this dataset.")