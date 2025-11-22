import numpy as np
import matplotlib.pyplot as plt


def binary_to_bipolar(pattern):
    return np.where(pattern == 0, -1, 1)

def bipolar_to_binary(pattern):
    return np.where(pattern == -1, 0, 1)


def train_hopfield(patterns):
    num_neurons = patterns[0].size
    weights = np.zeros((num_neurons, num_neurons))
    for pattern in patterns:
        pattern = pattern.reshape(-1, 1) 
        weights += np.dot(pattern, pattern.T)
    np.fill_diagonal(weights, 0)  
    return weights / len(patterns)


def recall(weights, input_pattern, max_iterations=100):
    num_neurons = input_pattern.size
    state = input_pattern.copy()
    for _ in range(max_iterations):
        for i in range(num_neurons):
            net_input = np.dot(weights[i], state)
            state[i] = 1 if net_input >= 0 else -1
    return state


def test_capacity():
    num_neurons = 100  
    max_patterns = 30  
    pattern_size = (10, 10)
    success_rates = []

    for num_patterns in range(1, max_patterns + 1):
       
        patterns = [binary_to_bipolar(np.random.randint(0, 2, pattern_size).flatten()) for _ in range(num_patterns)]
        
     
        weights = train_hopfield(patterns)
        
        
        successes = 0
        for pattern in patterns:
          
            noisy_pattern = pattern.copy()
            num_flips = int(0.1 * num_neurons)
            flip_indices = np.random.choice(num_neurons, num_flips, replace=False)
            noisy_pattern[flip_indices] *= -1 
            
           
            recalled_pattern = recall(weights, noisy_pattern)
            
           
            if np.array_equal(recalled_pattern, pattern):
                successes += 1
        
   
        success_rate = successes / num_patterns
        success_rates.append(success_rate)
    

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_patterns + 1), success_rates, marker='o')
    plt.axvline(x=int(0.15 * num_neurons), color='red', linestyle='--', label="Theoretical Capacity (15)")
    plt.title("Hopfield Network Capacity Test")
    plt.xlabel("Number of Stored Patterns")
    plt.ylabel("Recall Success Rate")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    test_capacity()