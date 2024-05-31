 import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Digits data set loading 
digits = load_digits()
X = digits.data
y = digits.target
target_names = digits.target_names

# initializing pca 
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)


# Variance for each component
print(f"Variance for each principal component: {pca.explained_variance_ratio_}")

# 2D vizualisation
plt.figure(figsize=(10, 8))
colors = plt.cm.get_cmap('tab10', 10) 

for i in range(len(target_names)):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=colors(i), alpha=0.8, label=f'Digit {i}')

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Digits dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()



