# Mobile Usage Clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans

np.random.seed(42)

data = pd.DataFrame({
    "ScreenTime": np.random.randint(1, 12, 60),
    "AppUsage": np.random.randint(5, 50, 60),
    "DataUsage": np.random.randint(100, 2000, 60)
})

print(data.head())

scaler = StandardScaler()
scaled = scaler.fit_transform(data)

for col in data.columns:
    plt.figure()
    sns.histplot(data[col], kde=True)
    plt.title(col)
    plt.show()

linked = linkage(scaled, method='ward')
plt.figure()
dendrogram(linked)
plt.title("Dendrogram")
plt.show()

agg = AgglomerativeClustering(n_clusters=3)
data["Cluster"] = agg.fit_predict(scaled)

sns.scatterplot(x=data["ScreenTime"], y=data["DataUsage"], hue=data["Cluster"])
plt.show()

print(data.groupby("Cluster").mean())

k = KMeans(n_clusters=3, random_state=42)
data["KMeans"] = k.fit_predict(scaled)

sns.scatterplot(x=data["ScreenTime"], y=data["DataUsage"], hue=data["KMeans"])
plt.show()