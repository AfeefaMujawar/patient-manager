# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ===============================
# SYNTHETIC DATASET (10 rows)
# ===============================
df = pd.DataFrame({
    'SepalLength':[5.1,4.9,5.8,6.0,6.3,6.5,5.5,6.7,5.0,6.2],
    'SepalWidth':[3.5,3.0,2.7,2.9,3.3,3.0,2.3,3.1,3.6,2.8],
    'PetalLength':[1.4,1.4,4.1,4.5,6.0,5.5,4.0,5.6,1.5,5.4],
    'PetalWidth':[0.2,0.2,1.3,1.5,2.5,2.0,1.3,2.4,0.2,2.3],
    'Species':['Setosa','Setosa','Versicolor','Versicolor','Virginica','Virginica','Versicolor','Virginica','Setosa','Virginica']
})

# ===============================
# PART A: SCALING + EDA
# ===============================
X = df.drop('Species', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Scatter relationships
plt.scatter(df['SepalLength'], df['PetalLength'])
plt.xlabel("SepalLength")
plt.ylabel("PetalLength")
plt.title("Feature Relationship")
plt.show()

pd.plotting.scatter_matrix(df.iloc[:,:4], figsize=(8,6))
plt.suptitle("Pair Plot")
plt.show()

# ===============================
# PART B: MODEL BUILDING
# ===============================

# Elbow Method
inertia = []
for k in range(1,6):
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1,6), inertia, marker='o')
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Optimal K = 3
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = labels

print("\nCluster Labels:\n", df[['Species','Cluster']])

# ===============================
# PART C: EVALUATION
# ===============================

# Visualization
plt.scatter(df['SepalLength'], df['PetalLength'], c=labels)
plt.xlabel("SepalLength")
plt.ylabel("PetalLength")
plt.title("K-Means Clusters")
plt.show()

# Compare with actual
print("\nCluster vs Actual:\n", pd.crosstab(df['Cluster'], df['Species']))

# Silhouette Score
score = silhouette_score(X_scaled, labels)
print("\nSilhouette Score:", score)

# ===============================
# INTERPRETATION
# ===============================
print("\nInterpretation:")
print("Clusters separate species based on petal features.")
print("Setosa forms a clear distinct cluster.")
print("Versicolor and Virginica show slight overlap.")