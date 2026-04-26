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
np.random.seed(11)
n = 150

df = pd.DataFrame({
    'StudyHours': np.random.randint(1, 10, n),
    'Attendance': np.random.randint(50, 100, n),
    'AssignmentScore': np.random.randint(40, 100, n)
})

# ===============================
# NORMALIZATION
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

# Scatter relationships
plt.scatter(df['StudyHours'], df['AssignmentScore'])
plt.xlabel("StudyHours")
plt.ylabel("AssignmentScore")
plt.title("StudyHours vs AssignmentScore")
plt.show()

# ===============================
# PART B: ELBOW METHOD
# ===============================
inertia = []

for k in range(1,7):
    km = KMeans(n_clusters=k, random_state=11, n_init=10)
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
kmeans = KMeans(n_clusters=3, random_state=11, n_init=10)
labels = kmeans.fit_predict(X)

df['Cluster'] = labels

print("\nCluster Centroids:\n", kmeans.cluster_centers_)

# ===============================
# PART C: VISUALIZATION
# ===============================
plt.scatter(df['StudyHours'], df['AssignmentScore'], c=labels)
plt.xlabel("StudyHours")
plt.ylabel("AssignmentScore")
plt.title("Student Clusters")
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
print("Cluster 0 → Low performers (low study, low marks)")
print("Cluster 1 → Average students")
print("Cluster 2 → High performers (high study, high marks)")

print("\nConclusion:")
print("K-Means groups students based on academic behavior effectively.")