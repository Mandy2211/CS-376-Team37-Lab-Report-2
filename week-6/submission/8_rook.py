#!/usr/bin/env python3
"""
hopfield_lab.py

Implements:
1) Hopfield associative memory (10x10 = 100 neurons) with Hebbian learning
2) Capacity experiment + plot
3) Error-correction demonstration (image saved)
4) Eight-Rook solver using greedy energy descent
5) 10-city TSP solver using Hopfield-style energy + greedy swap descent

Outputs (saved in ./hopfield_lab_outputs/):
- capacity_curve.png
- recall_example.png
- eight_rook_solution.png
- tsp_state_matrix.png
- summary_results.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd

# ---------- Configuration ----------
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

# ---------- Utility Functions ----------
def bin_to_bipolar(arr):
    return np.where(arr == 0, -1, 1)

def bipolar_to_bin(arr):
    return np.where(arr < 0, 0, 1)

def hamming(a, b):
    return np.sum(a != b)

# ---------- Eight-Rook Energy & Greedy Descent ----------
def eight_rook_energy(V, A=10.0, B=10.0, C=1.0):
    # V: 8x8 binary (0/1)
    row_sums = V.sum(axis=1)
    col_sums = V.sum(axis=0)
    E = A * np.sum((row_sums - 1.0) ** 2) + B * np.sum((col_sums - 1.0) ** 2)
    E += C * np.sum(V * (1 - V))
    return float(E)

def greedy_descent_eight_rooks(init_V=None, max_iters=5000, A=30.0, B=30.0, C=1.0):
    if init_V is None:
        V = np.random.choice([0, 1], size=(8, 8))
    else:
        V = init_V.copy()
    E = eight_rook_energy(V, A, B, C)
    for it in range(max_iters):
        i, j = np.random.randint(0, 8), np.random.randint(0, 8)
        V2 = V.copy()
        V2[i, j] = 1 - V2[i, j]
        E2 = eight_rook_energy(V2, A, B, C)
        if E2 < E:
            V = V2
            E = E2
        # early exit if perfect
        if np.all(V.sum(axis=1) == 1) and np.all(V.sum(axis=0) == 1):
            break
    return V, E, it


def main():

    # Eight-Rook
    print("Solving Eight-Rook problem...")
    V_sol, E_sol, it_used = greedy_descent_eight_rooks(max_iters=5000, A=30.0, B=30.0, C=1.0)
    print("Eight-Rook energy:", E_sol, "iterations:", it_used)
    print("Row sums:", V_sol.sum(axis=1))
    print("Col sums:", V_sol.sum(axis=0))
    plt.figure(figsize=(4, 4))
    plt.imshow(V_sol, interpolation='nearest')
    plt.title("Eight-Rook solution (1 = rook)")
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, "eight_rook_solution.png"))
    plt.close()
    print("Saved:", os.path.join(OUTPUT_DIR, "eight_rook_solution.png"))


if __name__ == "__main__":
    main()
