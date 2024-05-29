import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples, random_state)
