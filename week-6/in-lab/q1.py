"""
hopfield_lab.py

Implements:
1) Hopfield associative memory (10x10 = 100 neurons) with Hebbian learning
2) Capacity experiment + plot
3) Error-correction demonstration (image saved)

Outputs (saved in ./hopfield_lab_outputs/):
- capacity_curve.png
- recall_example.png
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

# ---------- Utility Functions ----------
def bin_to_bipolar(arr):
    return np.where(arr == 0, -1, 1)

def bipolar_to_bin(arr):
    return np.where(arr < 0, 0, 1)

def hamming(a, b):
    return np.sum(a != b)

# ---------- Capacity Test ----------
def capacity_test(N=100, trials=8, max_patterns=25, noise_frac=0.2):
    results = []
    for P in range(1, max_patterns + 1):
        success_count = 0
        for t in range(trials):
            patterns = np.random.choice([-1, 1], size=(P, N))
            net = HopfieldNetwork(N)
            net.train_hebbian(patterns, normalize=True)
            correct = 0
            for p in range(P):
                orig = patterns[p].copy()
                noisy = orig.copy()
                k = int(round(noise_frac * N))
                flip_idx = np.random.choice(N, size=k, replace=False)
                noisy[flip_idx] *= -1
                recalled = net.recall(noisy, steps=500)
                if np.array_equal(recalled, orig):
                    correct += 1
            if correct >= P * 0.9:
                success_count += 1
        results.append(success_count / trials)
    return results

# ---------- Main Script ----------
def main():
    # Part 1: capacity test
    print("Running capacity test...")
    cap_results = capacity_test(N=100, trials=8, max_patterns=25, noise_frac=0.2)
    plt.figure()
    plt.plot(range(1, len(cap_results) + 1), cap_results, marker='o')
    plt.xlabel("Number of stored patterns (P)")
    plt.ylabel("Fraction of trials with ≥90% recall")
    plt.title("Empirical capacity test (10x10 = 100 neurons)")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "capacity_curve.png"))
    plt.close()
    print("Saved:", os.path.join(OUTPUT_DIR, "capacity_curve.png"))

    # Part 2: error-correction demo
    N = 100
    P = 10
    patterns = np.random.choice([-1, 1], size=(P, N))
    net = HopfieldNetwork(N)
    net.train_hebbian(patterns)
    pat0 = patterns[0].copy()
    noisy = pat0.copy()
    k = int(0.25 * N)
    flip_idx = np.random.choice(N, size=k, replace=False)
    noisy[flip_idx] *= -1
    recalled = net.recall(noisy, steps=1000)
    hd_before = hamming(pat0, noisy)
    hd_after = hamming(pat0, recalled)
    print(f"Error-correction demo: Hamming before={hd_before}, after={hd_after}")

    # visualize patterns as 10x10 images
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow((pat0.reshape(10, 10) > 0).astype(int), interpolation='nearest')
    axes[0].set_title("Original")
    axes[1].imshow((noisy.reshape(10, 10) > 0).astype(int), interpolation='nearest')
    axes[1].set_title("Noisy (25% flips)")
    axes[2].imshow((recalled.reshape(10, 10) > 0).astype(int), interpolation='nearest')
    axes[2].set_title("Recalled")
    for ax in axes:
        ax.axis('off')
    plt.suptitle("Hopfield recall: original → noisy → recalled")
    plt.savefig(os.path.join(OUTPUT_DIR, "recall_example.png"))
    plt.close()
    print("Saved:", os.path.join(OUTPUT_DIR, "recall_example.png"))

if __name__ == "__main__":
    main()
