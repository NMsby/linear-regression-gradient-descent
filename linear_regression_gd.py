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


def compute_cost(X, y, slope, bias):
    """
    Compute the Mean Squared Error cost function.

    Parameters:
    -----------
    X: ndarray of shape (n_samples,)
        Input features
    y : ndarray of shape (n_samples,)
        Target values
    slope : float
        Current slope value
    bias : float
        Current bias value

    Returns:
    --------
    cost : float
        Mean Squared Error
    """
    n_samples = len(X)

    # Predictions using current parameters
    y_pred = bias + slope * X

    # Calculate MSE
    cost = np.sum((y_pred - y) ** 2) / (2 * n_samples)

    return cost


def compute_gradients(X, y, slope, bias):
    """
    Compute gradients of the cost function with respect to slope and bias.

    Parameters:
    -----------
    X : ndarray of shape (n_samples,)
        Input features
    y : ndarray of shape (n_samples,)
        Target values
    slope : float
        Current slope value
    bias : float
        Current bias value

    Returns:
    --------
    d_slope : float
        Gradient with respect to slope
    d_bias : float
        Gradient with respect to bias
    """
    n_samples = len(X)

    # Predictions using current parameters
    y_pred = bias + slope * X

    # Calculate gradients
    d_slope = np.sum((y_pred - y) * X) / n_samples
    d_bias = np.sum(y_pred - y) / n_samples

    return d_slope, d_bias


def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Implement gradient descent to find optimal slope and bias.

    Parameters:
    -----------
    X : ndarray of shape (n_samples,)
        Input features
    y : ndarray of shape (n_samples,)
        Target values
    learning_rate : float
        Step size for gradient descent updates
    n_iterations : int
        Number of iterations to run

    Returns:
    --------
    slope : float
        Optimal slope value
    bias : float
        Optimal bias value
    cost_history : list
        History of cost values during optimization
    slope_history : list
        History of slope values during optimization
    bias_history : list
        History of bias values during optimization
    """
    # Initialize parameters
    slope = 0.0
    bias = 0.0

    # Initialize lists to store values during optimization
    cost_history = []
    slope_history = []
    bias_history = []

    # Gradient descent iterations
    for i in range(n_iterations):
        # Compute current cost
        current_cost = compute_cost(X, y, slope, bias)
        cost_history.append(current_cost)
        slope_history.append(slope)
        bias_history.append(bias)

        # Compute gradients
        d_slope, d_bias = compute_gradients(X, y, slope, bias)

        # Update parameters
        slope = slope - learning_rate * d_slope
        bias = bias - learning_rate * d_bias

        # Optional: Print progress every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}/{n_iterations}, Cost: {current_cost:.4f}, Slope: {slope:.4f}, Bias: {bias:.4f}")

    # Compute final cost
    final_cost = compute_cost(X, y, slope, bias)

    return slope, bias, cost_history, slope_history, bias_history


def experiment_learning_rates(X, y, learning_rates=[0.001, 0.01, 0.05, 0.1]):
    """
    Experiment with different learning rates and compare the results.

    Parameters:
    -----------
    X : ndarray of shape (n_samples,)
        Input features
    y : ndarray of shape (n_samples,)
        Target values
    learning_rates : list
        List of learning rates to try
    """
    plt.figure(figsize=(15, 10))

    results = []

    for i, lr in enumerate(learning_rates):
        # Run gradient descent with current learning rate
        slope, bias, cost_history, _, _ = gradient_descent(X, y, learning_rate=lr, n_iterations=1000)
        results.append((lr, slope, bias, cost_history[-1]))

        # Plot cost history
        plt.subplot(2, 2, i + 1)
        plt.plot(cost_history)
        plt.title(f'Learning Rate: {lr}')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Compare final results
    print("\nComparison of Learning Rates:")
    print("-" * 60)
    print(f"{'Learning Rate':^15}{'Slope':^15}{'Bias':^15}{'Final MSE':^15}")
    print("-" * 60)
    for lr, slope, bias, mse in results:
        print(f"{lr:^15.4f}{slope:^15.4f}{bias:^15.4f}{mse:^15.4f}")
    print("-" * 60)


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