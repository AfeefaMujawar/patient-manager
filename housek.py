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
# SYNTHETIC DATASET (10 rows)
# ===============================
data = {
    'Area': [600, 800, 1200, 1500, 2000, 550, 900, 1300, 1800, 2200],
    'Bedrooms': [1, 2, 2, 3, 4, 1, 2, 3, 3, 4],
    'Price': [300000, 400000, 600000, 750000, 1000000, 280000, 420000, 650000, 900000, 1100000]
}
df = pd.DataFrame(data)

# ===============================
# PART A: SCALING + EDA
# ===============================
print("\nSummary:\n", df.describe())

# Feature relationships
plt.scatter(df['Area'], df['Price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()

# Distributions
df.hist()
plt.show()

# ===============================
# SCALING
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ===============================
# PART B: ELBOW METHOD
# ===============================
inertia = []

for k in range(1,6):
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1,6), inertia)
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Optimal K = 3

# ===============================
# K-MEANS MODEL
# ===============================
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = labels

print("\nClustered Data:\n", df)

# ===============================
# PART C: VISUALIZATION
# ===============================
plt.scatter(df['Area'], df['Price'], c=labels)
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Housing Clusters")
plt.show()

# ===============================
# EVALUATION
# ===============================
sil_score = silhouette_score(X_scaled, labels)

print("\nSilhouette Score:", sil_score)
print("\nCluster Counts:\n", df['Cluster'].value_counts())

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Cluster 0 → Low-cost houses (small area, fewer bedrooms)")
print("Cluster 1 → Mid-range houses")
print("Cluster 2 → Premium houses (large area, high price)")

print("\nConclusion:")
print("K-Means effectively segments houses into budget, mid-range, and luxury categories.")