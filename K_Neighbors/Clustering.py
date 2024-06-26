import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#iris = datasets.load_iris()
#X = iris.data
PCA_active = True 

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples, random_state)

n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

if PCA_active:
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

#Training
kmeans.fit(X)

"""
# DBSCAN Model
dbscan = DBSCAN(eps=0.3, min_samples=10)

# Training
y_pred = dbscan.fit_predict(X)
"""

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

"""
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples, random_state)
n_clusters_list = [2, 3, 4, 5]

kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

#Training
kmeans.fit(X)

#Prediction
y_pred = kmeans.predict(X)


for n_clusters in n_clusters_list:
    # Initialisation du modèle KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    # Entraînement du modèle
    kmeans.fit(X)

    # Prédiction des clusters
    y_pred = kmeans.predict(X)

    # Affichage des résultats
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
    plt.title(f'Clustering K-Means avec {n_clusters} clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

"""



# Data
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

#Distance with the mean of each clusters 
wcss = []
for n_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Elbow method
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow method to determine the number of clusters ')
plt.xlabel('Numbers of clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()



