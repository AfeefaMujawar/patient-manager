# =========================================
# Q13: Employee Grouping (Agglomerative)
# =========================================

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
    "Experience": np.random.randint(1, 20, 60),         # years
    "Salary": np.random.randint(20000, 120000, 60),     # yearly salary
    "PerformanceScore": np.random.randint(1, 10, 60)    # rating 1-10
})

print("\n===== HR DATASET =====")
print(data.head())

# Normalize Data
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
sns.histplot(data["Experience"], kde=True)
plt.title("Experience Distribution")
plt.show()

plt.figure()
sns.histplot(data["Salary"], kde=True)
plt.title("Salary Distribution")
plt.show()

plt.figure()
sns.histplot(data["PerformanceScore"], kde=True)
plt.title("Performance Score Distribution")
plt.show()

# ------------------------------
# Part B: Agglomerative Clustering
# ------------------------------

from scipy.cluster.hierarchy import dendrogram, linkage

# Linkage matrix
linked = linkage(scaled_data, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Dendrogram (Employee Grouping)")
plt.xlabel("Employees")
plt.ylabel("Distance")
plt.show()

# Apply Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

# Assume 3 clusters (based on dendrogram)
agglo = AgglomerativeClustering(n_clusters=3)
labels = agglo.fit_predict(scaled_data)

data["Cluster"] = labels

print("\n===== AGGLOMERATIVE CLUSTERS =====")
print(data.head())

# ------------------------------
# Part C: Visualization
# ------------------------------

plt.figure()
sns.scatterplot(x=data["Experience"], y=data["Salary"], hue=data["Cluster"], palette="Set1")
plt.title("Employee Groups (Agglomerative)")
plt.xlabel("Experience")
plt.ylabel("Salary")
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
sns.scatterplot(x=data["Experience"], y=data["Salary"], hue=data["KMeans_Cluster"], palette="Set2")
plt.title("K-Means Employee Groups")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# Final Output
print("\n===== FINAL DATA =====")
print(data.head())