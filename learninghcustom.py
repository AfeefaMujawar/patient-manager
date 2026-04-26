# Learning Pattern

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans

np.random.seed(42)

data = pd.DataFrame({
    "TimeSpent": np.random.randint(1,10,60),
    "QuizScore": np.random.randint(40,100,60),
    "Assignments": np.random.randint(1,20,60)
})

scaled = StandardScaler().fit_transform(data)

for col in data.columns:
    sns.histplot(data[col], kde=True)
    plt.show()

dendrogram(linkage(scaled,'ward'))
plt.show()

data["Cluster"] = AgglomerativeClustering(3).fit_predict(scaled)

sns.scatterplot(x=data["TimeSpent"], y=data["QuizScore"], hue=data["Cluster"])
plt.show()

print(data.groupby("Cluster").mean())

data["KMeans"] = KMeans(3).fit_predict(scaled)
sns.scatterplot(x=data["TimeSpent"], y=data["QuizScore"], hue=data["KMeans"])
plt.show()