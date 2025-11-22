import numpy as np

def value_iteration(step_cost=0.02, gamma=0.9, epsilon=1e-4):
    rows, cols = 3, 4
    V = np.zeros((rows, cols))
    V[0, 3] = 1
    V[1, 3] = -1

    actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    while True:
        delta = 0
        V_old = V.copy()

        for r in range(rows):
            for c in range(cols):
                if (r == 0 and c == 3) or (r == 1 and c == 3) or (r == 1 and c == 1):
                    continue

                action_values = []
                for action in actions:
                    action_value = 0

                    intended_r = r + action[0]
                    intended_c = c + action[1]

                    if 0 <= intended_r < rows and 0 <= intended_c < cols and (intended_r, intended_c) != (1, 1):
                        action_value += 0.8 * (step_cost + gamma * V_old[intended_r, intended_c])
                    else:
                        action_value += 0.8 * (step_cost + gamma * V_old[r, c])

                    left_action = (action[1], -action[0])
                    left_r = r + left_action[0]
                    left_c = c + left_action[1]

                    if 0 <= left_r < rows and 0 <= left_c < cols and (left_r, left_c) != (1, 1):
                        action_value += 0.1 * (step_cost + gamma * V_old[left_r, left_c])
                    else:
                        action_value += 0.1 * (step_cost + gamma * V_old[r, c])

                    right_action = (-action[1], action[0])
                    right_r = r + right_action[0]
                    right_c = c + right_action[1]

                    if 0 <= right_r < rows and 0 <= right_c < cols and (right_r, right_c) != (1, 1):
                        action_value += 0.1 * (step_cost + gamma * V_old[right_r, right_c])
                    else:
                        action_value += 0.1 * (step_cost + gamma * V_old[r, c])

                    action_values.append(action_value)

                V[r, c] = max(action_values)
                delta = max(delta, abs(V[r, c] - V_old[r, c]))

        if delta < epsilon:
            break

    return V


rewards = [-2, 0.1, 0.02, 1]

for r in rewards:
    print(f"\nValue Function for r(s) = {r}:")
    V = value_iteration(step_cost=r)
    print(V)
