# ==============================
# Q11: Student Behavior Clustering
# ==============================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fix for graph display (if needed)
import matplotlib
matplotlib.use('TkAgg')

# ------------------------------
# Part A: Dataset + Preprocessing
# ------------------------------

# Generate Dataset
np.random.seed(42)

data = pd.DataFrame({
    "StudyHours": np.random.randint(1, 10, 50),
    "Attendance": np.random.randint(50, 100, 50),
    "Marks": np.random.randint(40, 100, 50)
})

print("\n===== DATASET SAMPLE =====")
print(data.head())

# Normalize Features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

print("\n===== NORMALIZED DATA =====")
print(scaled_df.head())

# ------------------------------
# EDA: Distribution Plots
# ------------------------------

plt.figure()
sns.histplot(data["StudyHours"], kde=True)
plt.title("Study Hours Distribution")
plt.show()

plt.figure()
sns.histplot(data["Attendance"], kde=True)
plt.title("Attendance Distribution")
plt.show()

plt.figure()
sns.histplot(data["Marks"], kde=True)
plt.title("Marks Distribution")
plt.show()

# ------------------------------
# Part B: Agglomerative Clustering
# ------------------------------

from scipy.cluster.hierarchy import dendrogram, linkage

# Create linkage matrix
linked = linkage(scaled_data, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Dendrogram (Find Optimal Clusters)")
plt.xlabel("Students")
plt.ylabel("Distance")
plt.show()

# Apply Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

# Choose clusters (based on dendrogram, assume 3)
agglo = AgglomerativeClustering(n_clusters=3)
labels = agglo.fit_predict(scaled_data)

data["Agglo_Cluster"] = labels

print("\n===== AGGLOMERATIVE CLUSTERED DATA =====")
print(data.head())

# ------------------------------
# Part C: Visualization
# ------------------------------

plt.figure()
sns.scatterplot(x=data["StudyHours"], y=data["Marks"], hue=data["Agglo_Cluster"], palette="Set1")
plt.title("Agglomerative Clustering (Students)")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()

# Cluster Interpretation
print("\n===== CLUSTER MEANS (Agglomerative) =====")
print(data.groupby("Agglo_Cluster").mean())

# ------------------------------
# K-Means Comparison
# ------------------------------

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
k_labels = kmeans.fit_predict(scaled_data)

data["KMeans_Cluster"] = k_labels

# Plot KMeans clusters
plt.figure()
sns.scatterplot(x=data["StudyHours"], y=data["Marks"], hue=data["KMeans_Cluster"], palette="Set2")
plt.title("K-Means Clustering (Comparison)")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()

# ------------------------------
# Final Output
# ------------------------------

print("\n===== FINAL DATA WITH BOTH CLUSTERS =====")
print(data.head())