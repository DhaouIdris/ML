#f(x) = x^2

class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:      
        for i in range(iterations):
            init = init - learning_rate*2*init 
        return round(init*100000)/100000  





import numpy as np
from numpy.typing import NDArray

class Solution:
    
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        pred = np.matmul(X, weights)
        return np.round(pred, 5)


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        error = 0
        n = len(model_prediction)
        for i in range(n):
            error+= np.square(model_prediction[i][0]-ground_truth[i][0])
        return round(error/n,5)
