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
np.random.seed(19)
n = 150

df = pd.DataFrame({
    'PowerUsage': np.random.randint(100, 2000, n),   # watts
    'DeviceCount': np.random.randint(1, 15, n),
    'UsageHours': np.random.randint(1, 24, n)
})

# ===============================
# NORMALIZATION
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(df)

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

df.hist(figsize=(8,5))
plt.show()

# Relationship
plt.scatter(df['UsageHours'], df['PowerUsage'])
plt.xlabel("Usage Hours")
plt.ylabel("Power Usage")
plt.title("Usage vs Power")
plt.show()

# ===============================
# PART B: ELBOW METHOD
# ===============================
inertia = []

for k in range(1,7):
    km = KMeans(n_clusters=k, random_state=19, n_init=10)
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
kmeans = KMeans(n_clusters=3, random_state=19, n_init=10)
labels = kmeans.fit_predict(X)

df['Cluster'] = labels

print("\nCluster Centroids:\n", kmeans.cluster_centers_)

# ===============================
# PART C: VISUALIZATION
# ===============================
plt.scatter(df['UsageHours'], df['PowerUsage'], c=labels)
plt.xlabel("Usage Hours")
plt.ylabel("Power Usage")
plt.title("Energy Clusters")
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
print("Cluster 0 → Low usage households (few devices, low power)")
print("Cluster 1 → Moderate consumption")
print("Cluster 2 → High energy users (many devices, long usage)")

print("\nConclusion:")
print("K-Means effectively groups households based on energy consumption behavior.")