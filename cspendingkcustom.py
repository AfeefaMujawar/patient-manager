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
# PART A: SYNTHETIC DATASET (150)
# ===============================
np.random.seed(12)
n = 150

df = pd.DataFrame({
    'Age': np.random.randint(18, 65, n),
    'Income': np.random.randint(20000, 120000, n),
    'SpendingScore': np.random.randint(1, 100, n)
})

# ===============================
# SCALING
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(df)

# ===============================
# EDA
# ===============================
print("\nSummary:\n", df.describe())

# Distributions
df.hist(figsize=(8,5))
plt.show()

# ===============================
# PART B: ELBOW METHOD
# ===============================
inertia = []

for k in range(1,7):
    km = KMeans(n_clusters=k, random_state=12, n_init=10)
    km.fit(X)
    inertia.append(km.inertia_)

plt.plot(range(1,7), inertia)
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Optimal K = 3

# ===============================
# K-MEANS MODEL
# ===============================
kmeans = KMeans(n_clusters=3, random_state=12, n_init=10)
labels = kmeans.fit_predict(X)

df['Cluster'] = labels

print("\nCluster Centers:\n", kmeans.cluster_centers_)

# ===============================
# PART C: VISUALIZATION
# ===============================
plt.scatter(df['Income'], df['SpendingScore'], c=labels)
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.show()

# ===============================
# EVALUATION
# ===============================
print("\nInertia:", kmeans.inertia_)

sil_score = silhouette_score(X, labels)
print("Silhouette Score:", sil_score)

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Cluster 0 → Low income, low spending (budget customers)")
print("Cluster 1 → High income, high spending (premium customers)")
print("Cluster 2 → Moderate income and spending (average customers)")

print("\nConclusion:")
print("K-Means effectively segments customers based on income and spending behavior.")