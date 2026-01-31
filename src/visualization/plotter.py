import matplotlib.pyplot as plt
import os
import numpy as np

def plot_convergence(history, filename="convergence.png"):
    """
    Plots the convergence graph of the PSO algorithm.
    Adaptive scaling: log if positive, linear if negative values exist.
    """
    history = np.array(history)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Global Best Cost')
    plt.title('PSO Convergence History')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    
    # Adaptive Scaling: If any value is <= 0, we can't use Log scale
    if np.all(history > 1e-6):
        plt.yscale('log')
        
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
