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
    'StudyHours': [2, 3, 5, 6, 8, 1, 4, 7, 9, 2],
    'Attendance': [60, 65, 70, 75, 90, 55, 68, 85, 95, 58],
    'Marks':      [50, 55, 65, 70, 85, 45, 60, 78, 92, 48]
}
df = pd.DataFrame(data)

# ===============================
# PART A: CLEANING + EDA
# ===============================

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

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
plt.scatter(df['StudyHours'], df['Marks'], c=labels)
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Clusters")
plt.show()

# ===============================
# EVALUATION
# ===============================
sil_score = silhouette_score(X_scaled, labels)
print("\nSilhouette Score:", sil_score)

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Cluster 0 → Low performers (low study, low attendance, low marks)")
print("Cluster 1 → Average students (moderate study & marks)")
print("Cluster 2 → High performers (high study, high attendance, high marks)")

print("\nConclusion:")
print("K-Means successfully groups students based on academic behavior.")