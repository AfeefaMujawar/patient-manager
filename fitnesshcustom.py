# Fitness Clustering

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans

np.random.seed(42)

data = pd.DataFrame({
    "Steps": np.random.randint(1000,15000,60),
    "Calories": np.random.randint(1500,3500,60),
    "WorkoutTime": np.random.randint(10,120,60)
})

scaler = StandardScaler()
scaled = scaler.fit_transform(data)

for col in data.columns:
    sns.histplot(data[col], kde=True)
    plt.show()

linked = linkage(scaled,'ward')
dendrogram(linked)
plt.show()

data["Cluster"] = AgglomerativeClustering(3).fit_predict(scaled)

sns.scatterplot(x=data["Steps"], y=data["Calories"], hue=data["Cluster"])
plt.show()

print(data.groupby("Cluster").mean())

data["KMeans"] = KMeans(3).fit_predict(scaled)
sns.scatterplot(x=data["Steps"], y=data["Calories"], hue=data["KMeans"])
plt.show()