""" hopfield_lab.py

Implements:
5) 10-city TSP solver using Hopfield-style energy + greedy swap descent

Outputs (saved in ./hopfield_lab_outputs/):
- tsp_state_matrix.png
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd

OUTPUT_DIR = "hopfield_lab_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Hopfield Network Class ----------
class HopfieldNetwork:
    def __init__(self, N):
        self.N = N
        self.W = np.zeros((N, N))
    
    def train_hebbian(self, patterns, normalize=True):
        """
        patterns: array-like shape (P, N) with values in {-1, +1}
        """
        N = self.N
        W = np.zeros((N, N), dtype=float)
        for p in patterns:
            W += np.outer(p, p)
        np.fill_diagonal(W, 0)
        if normalize:
            W = W / N
        self.W = W
    
    def energy(self, x):
        return -0.5 * x.T @ self.W @ x
    
    def update_async(self, x_init, max_steps=1000):
        x = x_init.copy()
        N = self.N
        for step in range(max_steps):
            i = np.random.randint(0, N)
            raw = self.W[i] @ x
            x_new = 1 if raw >= 0 else -1
            x[i] = x_new
        return x
    
    def recall(self, pattern, steps=1000):
        return self.update_async(pattern.copy(), max_steps=steps)

def bin_to_bipolar(arr):
    return np.where(arr == 0, -1, 1)

def bipolar_to_bin(arr):
    return np.where(arr < 0, 0, 1)

def hamming(a, b):
    return np.sum(a != b)

# ---------- TSP Energy & Greedy Swap Descent ----------
def tsp_energy(state, dist_mat, A=500.0, B=500.0, C=1.0):
    N = state.shape[0]
    pos_sums = state.sum(axis=0)
    city_sums = state.sum(axis=1)
    E1 = A * np.sum((pos_sums - 1.0) ** 2) + B * np.sum((city_sums - 1.0) ** 2)
    E3 = 0.0
    for j in range(N):
        jp = (j + 1) % N
        E3 += np.sum(dist_mat * np.outer(state[:, j], state[:, jp]))
    return float(E1 + C * E3)

def greedy_descent_tsp(N=10, dist_mat=None, max_iters=20000, A=500.0, B=500.0, C=1.0):
    if dist_mat is None:
        rng = np.random.RandomState(42)
        D = rng.rand(N, N)
        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)
        dist_mat = D
    state = np.zeros((N, N), dtype=int)
    perm = list(range(N))
    random.shuffle(perm)
    for j in range(N):
        state[perm[j], j] = 1
    E = tsp_energy(state, dist_mat, A, B, C)
    for it in range(max_iters):
        j1, j2 = np.random.choice(N, size=2, replace=False)
        i1 = np.where(state[:, j1] == 1)[0][0]
        i2 = np.where(state[:, j2] == 1)[0][0]
        new_state = state.copy()
        new_state[i1, j1] = 0
        new_state[i2, j2] = 0
        new_state[i1, j2] = 1
        new_state[i2, j1] = 1
        E2 = tsp_energy(new_state, dist_mat, A, B, C)
        if E2 < E:
            state = new_state
            E = E2
    tour = [int(np.where(state[:, j] == 1)[0][0]) for j in range(N)]
    total_dist = 0.0
    for j in range(N):
        i = tour[j]
        k = tour[(j + 1) % N]
        total_dist += dist_mat[i, k]
    return state, E, tour, total_dist, dist_mat

# ---------- Main Script ----------
def main():
    # Part 4: TSP
    print("Solving TSP (10 cities)...")
    state, E_final, tour, total_dist, dist_mat = greedy_descent_tsp(N=10, max_iters=20000, A=500.0, B=500.0, C=1.0)
    print("TSP energy:", E_final)
    print("Tour:", tour)
    print("Total (scaled) distance:", total_dist)
    plt.figure(figsize=(6, 3))
    plt.imshow(state, interpolation='nearest')
    plt.title(f"TSP state matrix; total_dist={total_dist:.4f}")
    plt.ylabel("City index")
    plt.xlabel("Position in tour")
    plt.colorbar()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsp_state_matrix.png"))
    plt.close()
    print("Saved:", os.path.join(OUTPUT_DIR, "tsp_state_matrix.png"))


if __name__ == "__main__":
    main()
