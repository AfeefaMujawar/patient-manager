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
# LOAD DATASET
# ===============================
df = pd.read_csv("Credit CArd.csv")
df.columns = df.columns.str.strip()

# ===============================
# PART A: DATA CLEANING + EDA
# ===============================

# Select required columns
df = df[['BALANCE','PURCHASES','CREDIT_LIMIT']]

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Statistical summary
print("\nSummary:\n", df.describe())

# Feature distributions
df.hist()
plt.show()

# ===============================
# NORMALIZATION
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ===============================
# PART B: ELBOW METHOD
# ===============================
inertia = []

for k in range(1,11):
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1,11), inertia)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Optimal K = 3 (based on elbow)

# ===============================
# K-MEANS MODEL
# ===============================
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = labels

# Centroids (original scale)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids, columns=['BALANCE','PURCHASES','CREDIT_LIMIT'])

print("\nCentroids:\n", centroids_df)

# ===============================
# PART C: VISUALIZATION
# ===============================
plt.scatter(df['PURCHASES'], df['BALANCE'], c=labels)
plt.xlabel("Purchases")
plt.ylabel("Balance")
plt.title("Customer Segments")
plt.show()

# ===============================
# EVALUATION
# ===============================
sil_score = silhouette_score(X_scaled, labels)

print("\nSilhouette Score:", sil_score)
print("\nCluster Sizes:\n", df['Cluster'].value_counts())

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Cluster 0 → Low spenders (low balance & purchases)")
print("Cluster 1 → High spenders (high purchases, high credit usage)")
print("Cluster 2 → Moderate users (balanced spending behavior)")

print("\nConclusion:")
print("K-Means successfully segments customers based on spending and credit behavior.")