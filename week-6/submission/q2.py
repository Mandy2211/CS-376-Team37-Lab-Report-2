import numpy as np

N = 10
A, B, C = 500, 500, 1
max_iterations = 1000

np.random.seed(42)
distances = np.random.randint(1, 100, (N, N))
np.fill_diagonal(distances, 0)

state = np.random.randint(0, 2, (N, N))

def energy(state):
    E1 = A * np.sum((np.sum(state, axis=1) - 1) ** 2)
    E2 = B * np.sum((np.sum(state, axis=0) - 1) ** 2)
    E3 = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                next_pos = (j + 1) % N
                E3 += C * distances[i, k] * state[i, j] * state[k, next_pos]
    return E1 + E2 + E3

def update(state):
    new_state = np.copy(state)
    for i in range(N):
        for j in range(N):
            input_sum = 0
            for k in range(N):
                for l in range(N):
                    next_pos = (l + 1) % N
                    input_sum += -A * (np.sum(state[i, :]) - 1)
                    input_sum += -B * (np.sum(state[:, j]) - 1)
                    input_sum += -C * distances[i, k] * state[k, next_pos]
            new_state[i, j] = 1 if input_sum < 0 else 0
    return new_state

for iteration in range(max_iterations):
    new_state = update(state)
    if np.array_equal(new_state, state):
        break
    state = new_state

def validate_tour(state):
    row_sums = np.sum(state, axis=1)
    col_sums = np.sum(state, axis=0)
    return np.all(row_sums == 1) and np.all(col_sums == 1)

valid = validate_tour(state)
print("Distance Matrix:")
print(distances)
print("\nFinal State (Tour Representation):")
print(state)
if valid:
    print("\nValid Tour Found!")
else:
    print("\nInvalid Tour.")