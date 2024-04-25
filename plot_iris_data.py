import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assuming the first script is named 'iris_analysis.py' and the functions are available for import.
from iris_analysis import analyze_iris_data, load_data


def plot_iris_data(file_path):
    # Load the data
    data = load_data(file_path)

    # Perform the analysis
    pearson_correlations, linear_regression_equations = analyze_iris_data(file_path)

    # Plot the scatter plots and regression lines
    fig, axs = plt.subplots(3, 2, figsize=(12, 18))  # 3x2 subplot grid
    axs = axs.flatten()  # Flatten to 1D array for easy iteration

    for ax, ((col1, col2), corr) in zip(axs, pearson_correlations.items()):
        # Scatter plot
        ax.scatter(data[col1], data[col2], color='blue')

        # Regression line
        # Extract the numerical parts of the equation (removing 'y = ')
        equation = linear_regression_equations[(col1, col2)].replace('y = ', '')
        slope, intercept = map(float, equation.split('x + '))
        x_values = np.array(ax.get_xlim())
        y_values = intercept + slope * x_values
        ax.plot(x_values, y_values, color='red')

        # Title and labels
        ax.set_title(f"{col1} vs {col2}\nr = {corr:.2f}; y = {slope:.2f}x + {intercept:.2f}")
        ax.set_xlabel(f"{col1} (cm)")
        ax.set_ylabel(f"{col2} (cm)")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # File path to the data.csv
    file_path = '../pythonProject3/data.csv'
    plot_iris_data(file_path)
