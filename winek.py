# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

# Scatter relationship
plt.scatter(df['Alcohol'], df['MalicAcid'])
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.title("Feature Relationship")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================

# Elbow Method
inertia = []
for k in range(1,6):
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1,6), inertia, marker='o')
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Optimal K = 3
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = labels

# Cluster centers
print("\nCluster Centers:\n", kmeans.cluster_centers_)

# ===============================
# PART C: EVALUATION
# ===============================

# Visualization
plt.scatter(df['Alcohol'], df['MalicAcid'], c=labels)
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.title("Wine Clusters")
plt.show()

# Silhouette Score
score = silhouette_score(X_scaled, labels)
print("\nSilhouette Score:", score)

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Cluster 0 → Moderate alcohol wines")
print("Cluster 1 → High alcohol & magnesium wines (premium)")
print("Cluster 2 → Low alcohol wines (lighter category)")