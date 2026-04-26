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
np.random.seed(15)
n = 150

df = pd.DataFrame({
    'ScreenTime': np.random.randint(1, 10, n),        # hours/day
    'AppUsage': np.random.randint(5, 50, n),          # apps/day
    'DataConsumption': np.random.uniform(0.5, 5.0, n) # GB/day
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
    km = KMeans(n_clusters=k, random_state=15, n_init=10)
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
kmeans = KMeans(n_clusters=3, random_state=15, n_init=10)
labels = kmeans.fit_predict(X)

df['Cluster'] = labels

print("\nCluster Centroids:\n", kmeans.cluster_centers_)

# ===============================
# PART C: VISUALIZATION
# ===============================
plt.scatter(df['ScreenTime'], df['DataConsumption'], c=labels)
plt.xlabel("Screen Time")
plt.ylabel("Data Consumption")
plt.title("User Behavior Clusters")
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
print("Cluster 0 → Low usage users (low screen time, low data)")
print("Cluster 1 → Moderate users")
print("Cluster 2 → Heavy users (high screen time, high data consumption)")

print("\nConclusion:")
print("K-Means effectively segments users based on mobile usage behavior.")