# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv("Wholesale customers data.csv")
df.columns = df.columns.str.strip()

# ===============================
# PART A: PREPROCESSING & EDA
# ===============================

# Select required features
df = df[['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']]

# Handle missing values
df = df.dropna()

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Feature Distribution
df.hist(figsize=(10,6))
plt.suptitle("Feature Distributions")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================

# Elbow Method
inertia = []
for k in range(1,11):
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1,11), inertia, marker='o')
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Choose optimal K (3 or 4)
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Add cluster to dataset
df['Cluster'] = labels

# Centroids
print("\nCluster Centroids:\n", kmeans.cluster_centers_)

# ===============================
# PART C: EVALUATION & VISUALIZATION
# ===============================

# Scatter Plot (Milk vs Grocery)
plt.scatter(df['Milk'], df['Grocery'], c=labels)
plt.xlabel("Milk")
plt.ylabel("Grocery")
plt.title("Customer Clusters")
plt.show()

# Silhouette Score
score = silhouette_score(X_scaled, labels)
print("\nSilhouette Score:", score)

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Cluster 0 → High Grocery & Detergents (Retail buyers)")
print("Cluster 1 → High Fresh & Frozen (Hotels/Restaurants)")
print("Cluster 2 → Moderate spenders (General customers)")