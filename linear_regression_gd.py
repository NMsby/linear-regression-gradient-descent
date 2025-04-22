# Linear Regression Assignment: Gradient Descent Implementation

# This script implements linear regression using gradient descent.

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)


def generate_dataset(n_samples=200, noise_level=10):
    """
    Generate synthetic data for linear regression.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise_level : float
        Standard deviation of the Gaussian noise

    Returns:
    --------
    x : ndarray of shape (n_samples,)
        Input features
    y : ndarray of shape (n_samples,)
        Target values
    """
    # True parameters (unknown to the model)
    true_bias = 5
    true_slope = 2

    # Generate x values within range [0, 10]
    x = np.random.uniform(0, 10, n_samples)

    # Generate y values with some noise
    y = true_bias + true_slope * x + np.random.normal(0, noise_level, n_samples)

    return x, y


def main():
    """Main function to run the linear regression implementation."""
    # Generate the dataset
    X, y = generate_dataset()

    # Visualize the data
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.7)
    plt.title('Generated Dataset')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()