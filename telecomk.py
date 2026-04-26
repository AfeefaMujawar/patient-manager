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
# SYNTHETIC DATASET (10 rows)
# ===============================
data = {
    'Tenure': [1, 3, 6, 12, 24, 2, 8, 15, 30, 36],
    'MonthlyCharges': [20, 30, 50, 60, 80, 25, 55, 65, 90, 100],
    'TotalCharges': [20, 90, 300, 720, 1920, 50, 440, 975, 2700, 3600]
}
df = pd.DataFrame(data)

# ===============================
# PART A: CLEANING + EDA
# ===============================
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary:\n", df.describe())

# Distribution
df.hist()
plt.show()

# Relationship
plt.scatter(df['Tenure'], df['TotalCharges'])
plt.xlabel("Tenure")
plt.ylabel("Total Charges")
plt.title("Tenure vs Total Charges")
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

for k in range(1,6):
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1,6), inertia)
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Optimal K = 3

# ===============================
# K-MEANS MODEL
# ===============================
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = labels

print("\nCluster Assignments:\n", df)

# ===============================
# PART C: VISUALIZATION
# ===============================
plt.scatter(df['Tenure'], df['TotalCharges'], c=labels)
plt.xlabel("Tenure")
plt.ylabel("Total Charges")
plt.title("Customer Clusters")
plt.show()

# ===============================
# EVALUATION
# ===============================
print("\nInertia:", kmeans.inertia_)
sil_score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", sil_score)

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Cluster 0 → New customers (low tenure, low charges)")
print("Cluster 1 → Mid-term customers (moderate usage)")
print("Cluster 2 → Long-term high-value customers")

print("\nConclusion:")
print("K-Means effectively segments telecom users based on usage and tenure.")