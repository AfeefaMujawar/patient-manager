import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Load dataset
df = pd.read_csv("Mall_Customers.csv")
df.columns = df.columns.str.strip()

# --- Part A ---
# Remove irrelevant column
df = df.drop(columns=['CustomerID'])

# Encode Gender
df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})

# Select features
X = df[['Annual Income (k$)','Spending Score (1-100)']]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Distribution
df[['Age','Annual Income (k$)','Spending Score (1-100)']].hist()
plt.show()

# Scatter
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Distribution")
plt.show()

# --- Part B ---
# Dendrogram
plt.figure(figsize=(14,6))

sch.dendrogram(
    sch.linkage(X_scaled, method='ward'),
    truncate_mode='lastp',
    p=10                   # last 10 clusters only
)

plt.title("Dendrogram (Last Clusters)")
plt.xlabel("Clusters")
plt.ylabel("Distance")
plt.show()

# 🔥 Choose K = 5 (based on dendrogram)
hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)

# --- Part C ---
# Visualization
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=labels_hc)
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Hierarchical Clusters")
plt.show()

# Evaluation
print("\nHierarchical Silhouette Score:", silhouette_score(X_scaled, labels_hc))

# --- Comparison with K-Means ---
kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

print("K-Means Silhouette Score:", silhouette_score(X_scaled, labels_km))

# Interpretation
print("\nInterpretation:")
print("Hierarchical clustering groups customers into segments based on income and spending.")
print("High-income high-spending customers form premium segments.")
print("K-Means gives similar clusters but is faster, while hierarchical shows clear structure via dendrogram.")