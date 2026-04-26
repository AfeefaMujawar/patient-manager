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
df = pd.read_csv("OnlineRetail.csv", encoding='latin1')
df.columns = df.columns.str.strip()

# ===============================
# PART A: CLEANING + AGGREGATION
# ===============================

# Remove missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Remove negative/zero values
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Convert CustomerID to int
df['CustomerID'] = df['CustomerID'].astype(int)

# Aggregate per customer
customer_df = df.groupby('CustomerID').agg({
    'Quantity':'sum',
    'UnitPrice':'mean'
}).reset_index()

print("\nAggregated Data:\n", customer_df.head())

# ===============================
# SCALING
# ===============================
X = customer_df[['Quantity','UnitPrice']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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

# Optimal K = 3
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels = kmeans.fit_predict(X_scaled)

customer_df['Cluster'] = labels

# Cluster centers
print("\nCluster Centers:\n", kmeans.cluster_centers_)

# ===============================
# PART C: EVALUATION
# ===============================

# Visualization
plt.scatter(customer_df['Quantity'], customer_df['UnitPrice'], c=labels)
plt.xlabel("Quantity")
plt.ylabel("Unit Price")
plt.title("Customer Clusters")
plt.show()

# Metrics
inertia_val = kmeans.inertia_
sil_score = silhouette_score(X_scaled, labels)

print("\nInertia:", inertia_val)
print("Silhouette Score:", sil_score)

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Cluster 0 → Low quantity, low spending customers")
print("Cluster 1 → High quantity buyers (bulk purchasers)")
print("Cluster 2 → Moderate buyers with balanced spending")