import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def standardize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std