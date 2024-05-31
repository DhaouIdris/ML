 import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


# Digits data set loading 
digits = load_digits()
X = digits.data
y = digits.target
target_names = digits.target_names

# initializing pca 
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)




