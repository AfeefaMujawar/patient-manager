# =========================================
# Vehicle Usage Clustering (Hierarchical)
# =========================================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fix for graphs (if needed)
import matplotlib
matplotlib.use('TkAgg')

# ------------------------------
# Part A: Dataset + Preprocessing
# ------------------------------

# Generate Dataset
np.random.seed(42)

data = pd.DataFrame({
    "Distance": np.random.randint(5, 200, 60),     # km per trip
    "FuelUsage": np.random.uniform(1, 20, 60),     # liters
    "Speed": np.random.randint(20, 120, 60)        # km/h
})

print("\n===== VEHICLE DATASET =====")
print(data.head())

# Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

print("\n===== NORMALIZED DATA =====")
print(scaled_df.head())

# ------------------------------
# EDA: Distribution Analysis
# ------------------------------

plt.figure()
sns.histplot(data["Distance"], kde=True)
plt.title("Distance Distribution")
plt.show()

plt.figure()
sns.histplot(data["FuelUsage"], kde=True)
plt.title("Fuel Usage Distribution")
plt.show()

plt.figure()
sns.histplot(data["Speed"], kde=True)
plt.title("Speed Distribution")
plt.show()

# ------------------------------
# Part B: Hierarchical Clustering
# ------------------------------

from scipy.cluster.hierarchy import dendrogram, linkage

# Linkage matrix
linked = linkage(scaled_data, method='ward')

# Dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Dendrogram (Vehicle Usage)")
plt.xlabel("Vehicles")
plt.ylabel("Distance")
plt.show()

# Apply Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

# Assume 3 clusters (based on dendrogram)
agglo = AgglomerativeClustering(n_clusters=3)
labels = agglo.fit_predict(scaled_data)

data["Cluster"] = labels

print("\n===== CLUSTERED DATA =====")
print(data.head())

# ------------------------------
# Part C: Visualization
# ------------------------------

plt.figure()
sns.scatterplot(x=data["Distance"], y=data["FuelUsage"], hue=data["Cluster"], palette="Set1")
plt.title("Vehicle Usage Clusters (Hierarchical)")
plt.xlabel("Distance")
plt.ylabel("Fuel Usage")
plt.show()

# Cluster Interpretation
print("\n===== CLUSTER MEANS =====")
print(data.groupby("Cluster").mean())

# ------------------------------
# Compare with K-Means
# ------------------------------

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
k_labels = kmeans.fit_predict(scaled_data)

data["KMeans_Cluster"] = k_labels

plt.figure()
sns.scatterplot(x=data["Distance"], y=data["FuelUsage"], hue=data["KMeans_Cluster"], palette="Set2")
plt.title("K-Means Vehicle Clusters")
plt.xlabel("Distance")
plt.ylabel("Fuel Usage")
plt.show()

# Final Output
print("\n===== FINAL DATA =====")
print(data.head())