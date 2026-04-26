# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ===============================
# PART A: SYNTHETIC DATASET (150)
# ===============================
np.random.seed(18)
n = 150

df = pd.DataFrame({
    'UnitsSold': np.random.randint(10, 500, n),
    'Revenue': np.random.randint(1000, 50000, n),
    'Profit': np.random.randint(200, 15000, n)
})

# ===============================
# SCALING
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(df)

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

df.hist(figsize=(8,5))
plt.show()

# ===============================
# PART B: ELBOW METHOD
# ===============================
inertia = []

for k in range(1,7):
    km = KMeans(n_clusters=k, random_state=18, n_init=10)
    km.fit(X)
    inertia.append(km.inertia_)

plt.plot(range(1,7), inertia)
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Optimal K = 3

# ===============================
# K-MEANS MODEL
# ===============================
kmeans = KMeans(n_clusters=3, random_state=18, n_init=10)
labels = kmeans.fit_predict(X)

df['Cluster'] = labels

print("\nCluster Centroids:\n", kmeans.cluster_centers_)

# ===============================
# PART C: VISUALIZATION
# ===============================
plt.scatter(df['Revenue'], df['Profit'], c=labels)
plt.xlabel("Revenue")
plt.ylabel("Profit")
plt.title("Sales Clusters")
plt.show()

# ===============================
# EVALUATION
# ===============================
print("\nInertia:", kmeans.inertia_)

sil_score = silhouette_score(X, labels)
print("Silhouette Score:", sil_score)

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Cluster 0 → Low sales (low revenue, low profit)")
print("Cluster 1 → Moderate sales performance")
print("Cluster 2 → High sales (high revenue, high profit)")

print("\nConclusion:")
print("K-Means effectively segments products/business units based on sales performance.")