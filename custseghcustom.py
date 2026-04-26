# =========================================
# Q12: Customer Segmentation (Hierarchical)
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
    "Age": np.random.randint(18, 60, 60),
    "Income": np.random.randint(20000, 100000, 60),
    "SpendingScore": np.random.randint(1, 100, 60)
})

print("\n===== CUSTOMER DATASET =====")
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
sns.histplot(data["Age"], kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure()
sns.histplot(data["Income"], kde=True)
plt.title("Income Distribution")
plt.show()

plt.figure()
sns.histplot(data["SpendingScore"], kde=True)
plt.title("Spending Score Distribution")
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
plt.title("Dendrogram (Customer Segmentation)")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

# Apply Clustering
from sklearn.cluster import AgglomerativeClustering

# Assume 3 clusters from dendrogram
agglo = AgglomerativeClustering(n_clusters=3)
labels = agglo.fit_predict(scaled_data)

data["Cluster"] = labels

print("\n===== CLUSTERED DATA =====")
print(data.head())

# ------------------------------
# Part C: Visualization
# ------------------------------

plt.figure()
sns.scatterplot(x=data["Income"], y=data["SpendingScore"], hue=data["Cluster"], palette="Set1")
plt.title("Customer Segments (Hierarchical)")
plt.xlabel("Income")
plt.ylabel("Spending Score")
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
sns.scatterplot(x=data["Income"], y=data["SpendingScore"], hue=data["KMeans_Cluster"], palette="Set2")
plt.title("K-Means Customer Segments")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.show()

# Final Output
print("\n===== FINAL DATA =====")
print(data.head())