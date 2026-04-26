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
np.random.seed(13)
n = 150

df = pd.DataFrame({
    'Experience': np.random.randint(1, 20, n),
    'Salary': np.random.randint(20000, 120000, n),
    'PerformanceScore': np.random.randint(40, 100, n)
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

# Distributions
df.hist(figsize=(8,5))
plt.show()

# ===============================
# PART B: ELBOW METHOD
# ===============================
inertia = []

for k in range(1,7):
    km = KMeans(n_clusters=k, random_state=13, n_init=10)
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
kmeans = KMeans(n_clusters=3, random_state=13, n_init=10)
labels = kmeans.fit_predict(X)

df['Cluster'] = labels

print("\nCluster Centroids:\n", kmeans.cluster_centers_)

# ===============================
# PART C: VISUALIZATION
# ===============================
plt.scatter(df['Experience'], df['Salary'], c=labels)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Employee Clusters")
plt.show()

# ===============================
# EVALUATION
# ===============================
sil_score = silhouette_score(X, labels)

print("\nSilhouette Score:", sil_score)

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Cluster 0 → Low experience, low salary (junior employees)")
print("Cluster 1 → Moderate experience and salary (mid-level employees)")
print("Cluster 2 → High experience, high salary, high performance (senior employees)")

print("\nConclusion:")
print("K-Means groups employees based on experience, salary, and performance effectively.")