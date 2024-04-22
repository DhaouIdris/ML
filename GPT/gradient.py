#f(x) = x^2

class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:      
        for i in range(iterations):
            init = init - learning_rate*2*init 
        return round(init*100000)/100000  
