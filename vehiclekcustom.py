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
np.random.seed(14)
n = 150

df = pd.DataFrame({
    'DailyDistance': np.random.randint(5, 150, n),          # km
    'FuelConsumption': np.random.uniform(5, 20, n),         # L/100km
    'Speed': np.random.randint(30, 120, n)                  # km/h
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

# Distributions
df.hist(figsize=(8,5))
plt.show()

# ===============================
# PART B: ELBOW METHOD
# ===============================
inertia = []

for k in range(1,7):
    km = KMeans(n_clusters=k, random_state=14, n_init=10)
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
kmeans = KMeans(n_clusters=3, random_state=14, n_init=10)
labels = kmeans.fit_predict(X)

df['Cluster'] = labels

print("\nCluster Centroids:\n", kmeans.cluster_centers_)

# ===============================
# PART C: VISUALIZATION
# ===============================
plt.scatter(df['DailyDistance'], df['FuelConsumption'], c=labels)
plt.xlabel("Daily Distance")
plt.ylabel("Fuel Consumption")
plt.title("Driving Pattern Clusters")
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
print("Cluster 0 → Short distance, low fuel usage (efficient drivers)")
print("Cluster 1 → Moderate driving pattern")
print("Cluster 2 → Long distance, high fuel usage (heavy drivers)")

print("\nConclusion:")
print("K-Means identifies distinct driving patterns based on distance, speed, and fuel consumption.")