import numpy as np

class HopfieldNetwork:
    def __init__(self, num_units):
        self.num_units = num_units
        self.weights = np.zeros((num_units, num_units))
        
    def _bipolar(self, x):
        return np.where(x > 0, 1, -1)
    
    def train(self, patterns):
        for pattern in patterns:
            pattern = self._bipolar(pattern)
            self.weights += np.outer(pattern, pattern)
            
        np.fill_diagonal(self.weights, 0)
        
    def recall(self, input_pattern, steps=5):
        input_pattern = self._bipolar(input_pattern)
        for _ in range(steps):
            for i in range(self.num_units):
                net_input = np.dot(self.weights[i], input_pattern)
                input_pattern[i] = 1 if net_input > 0 else -1
                
        return input_pattern
    

if __name__ == "__main__":
    #create Hopfield network
    hopfield_net = HopfieldNetwork(num_units=4) #edit number of units here
    
    #define training patterns
    patterns = [
        np.array([1, 1, -1, -1]),
        np.array([1, -1, 1, -1])
    ]
    
    #train network 
    hopfield_net.train(patterns)
    
    #test with noisy pattern
    test_pattern = np.array([1, 1, -1, 1])
    
    recalled_pattern = hopfield_net.recall(test_pattern)
    
    print("Input Pattern:", test_pattern)
    print("Recalled Pattern:", recalled_pattern)
    
