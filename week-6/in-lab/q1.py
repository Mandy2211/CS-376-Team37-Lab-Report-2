import numpy as np


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


if __name__ == "__main__":
    patterns = [
        np.random.randint(0, 2, (10, 10)), 
        np.random.randint(0, 2, (10, 10)),  
        np.random.randint(0, 2, (10, 10)),  
    ]
    
    bipolar_patterns = [binary_to_bipolar(p.flatten()) for p in patterns]

    weights = train_hopfield(bipolar_patterns)
    

    test_pattern = patterns[0].copy()
    test_pattern[0][0] = 1 - test_pattern[0][0] 
    bipolar_test_pattern = binary_to_bipolar(test_pattern.flatten())
    

    recalled_pattern = recall(weights, bipolar_test_pattern)
    recalled_pattern = recalled_pattern.reshape(10, 10)
    binary_recalled_pattern = bipolar_to_binary(recalled_pattern)
    
    print("Original Pattern:")
    print(patterns[0])
    print("\nNoisy Input Pattern:")
    print(test_pattern)
    print("\nRecalled Pattern:")
    print(binary_recalled_pattern)