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
df = pd.read_csv("Social_Network_Ads.csv")
df.columns = df.columns.str.strip()

# Select required columns
df = df[['Age','EstimatedSalary']]

# ===============================
# PART A: SCALING + EDA
# ===============================

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Distribution
df.hist(figsize=(8,4))
plt.suptitle("Feature Distributions")
plt.show()

# Scatter plot
plt.scatter(df['Age'], df['EstimatedSalary'])
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("User Distribution")
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

# Optimal K = 3
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = labels

# ===============================
# PART C: EVALUATION
# ===============================

# Cluster Visualization
plt.scatter(df['Age'], df['EstimatedSalary'], c=labels)
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("User Clusters")
plt.show()

# Silhouette Score
score = silhouette_score(X_scaled, labels)
print("\nSilhouette Score:", score)

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Cluster 0 → Young, low income users")
print("Cluster 1 → Middle-aged, medium income users")
print("Cluster 2 → Older, high income users")