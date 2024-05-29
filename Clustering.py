import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples, random_state)

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

#Training
kmeans.fit(X)


y_pred = kmeans.predict(X)
