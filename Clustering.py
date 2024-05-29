import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#iris = datasets.load_iris()
#X = iris.data

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples, random_state)

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

#Training
kmeans.fit(X)

#Prediction
y_pred = kmeans.predict(X)

#Results
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('Résultats du clustering K-Means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()








