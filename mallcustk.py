import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Mall_Customers.csv")
df.columns = df.columns.str.strip()

# --- Part A ---
# Remove irrelevant column
df = df.drop(columns=['CustomerID'])

# Encode Gender
df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})

# Select features
X = df[['Age','Annual Income (k$)','Spending Score (1-100)']]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Distribution
df[['Age','Annual Income (k$)','Spending Score (1-100)']].hist()
plt.show()

# Scatter visualization
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Income vs Spending")
plt.show()

# --- Part B ---
# Elbow Method
inertia = []
K = range(1,11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# 🔥 Choose K = 5 (typical optimal)
kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
print("\nCluster Centroids:\n", pd.DataFrame(centroids, columns=X.columns))

# --- Part C ---
# Cluster visualization
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=labels)
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.show()

# Evaluation
print("\nInertia:", kmeans.inertia_)
print("Silhouette Score:", silhouette_score(X_scaled, labels))

# Interpretation
print("\nInterpretation:")
print("Customers are segmented into groups such as high-income high-spenders, low-income low-spenders, and moderate groups. This helps in targeted marketing strategies.")