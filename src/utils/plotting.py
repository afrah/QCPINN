from matplotlib import pyplot as plt
import matplotlib
import pylab as py
# from qiskit.visualization import plot_histogram

import numpy as np
import os
# Defining custom plotting functions
def plt_contourf(x, y, F, ttl):
    py.contourf(x, y, F, 41, cmap="inferno")
    py.colorbar()
    py.xlabel("x")
    py.ylabel("y")
    py.title(ttl)
    return 0


def plt_scatter(x, y, clr, ttl):
    py.plot(x, y, ".", markersize=2, color=clr)
    py.xlabel("x")
    py.ylabel("y")
    py.title(ttl)
    return 0


def plot_custom_histogram(sncounts1, width, height, file_name, font_size=10):
    # Create the histogram
    fig, ax = plt.subplots(figsize=(width, height))
    plot_histogram(sncounts1, ax=ax)  # Use default functionality to draw the histogram

    # Remove duplicate labels if automatically added
    for text in ax.texts:  # Remove all existing annotations
        text.set_visible(False)

    # Add custom labels (if necessary)
    for p in ax.patches:  # Iterate through bars
        height = p.get_height()
        ax.annotate(
            f"{height:.3f}",
            (p.get_x() + p.get_width() / 2.0, height),  # Position label on top
            ha="center",
            va="bottom",
            fontsize=font_size + 2,
        )  # Adjust font size

    # Customize axis labels
    ax.set_xlabel("States", fontsize=font_size + 4)
    ax.set_ylabel("Quasi-probability", fontsize=font_size + 4)

    # Adjust tick labels
    ax.tick_params(axis="x", labelsize=font_size + 2, rotation=45)
    ax.tick_params(axis="y", labelsize=font_size + 2)

    # Save and display
    plt.savefig("plot_histogram_identity.pdf", bbox_inches="tight")

def plot_results(u_analytical, u_approximate, x_axis, y_axis, title, file_name):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    
    # Plot analytical solution
    c1 = axs[0].contourf(x_axis, y_axis, u_analytical.reshape(x_axis.shape[0],
                                                              y_axis.shape[0]),
                                                              levels=50, cmap="inferno")
    axs[0].set_title(title[0], fontsize=14)
    axs[0].set_xlabel('x', fontsize=12)
    axs[0].set_ylabel('y', fontsize=12)
    axs[0].tick_params(axis='both', which='major', labelsize=10)
    cbar1 = fig.colorbar(c1, ax=axs[0], orientation='vertical', shrink=0.8, pad=0.02)
    cbar1.ax.tick_params(labelsize=10)

    # Plot approximate solution
    c2 = axs[1].contourf(x_axis, y_axis, u_approximate.reshape(x_axis.shape[0],
                                                               y_axis.shape[0]),
                                                               levels=50, cmap="inferno")
    axs[1].set_title(title[1], fontsize=14)
    axs[1].set_xlabel('x', fontsize=12)
    axs[1].set_ylabel('y', fontsize=12)
    axs[1].tick_params(axis='both', which='major', labelsize=10)
    cbar2 = fig.colorbar(c2, ax=axs[1], orientation='vertical', shrink=0.8 , pad=0.02 )
    cbar2.ax.tick_params(labelsize=10)

    # Save the figure
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)


def plot_exact_solution2(u, x_axis, y_axis, size, title, file_name):
    """
    Plot the exact solution of the Poisson equation.

    Args:
        u (array): Solution vector.
        x_axis (array): Spatial coordinates along the x-axis.
        y_axis (array): Spatial coordinates along the y-axis.
        size (int): Grid size (assumes square grid).
        title (list): Title for the plot (list with one element).
        file_name (str): Name of the file to save the plot.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(14, 4.8))

    # Plot the exact solution
    c1 = ax.contourf(
        x_axis, y_axis, abs(u.reshape(size, size)), levels=100, cmap="inferno"
    )
    fig.colorbar(c1, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title[0])

    # Save and display the plot
    plt.savefig(file_name, bbox_inches="tight")
    plt.close('all' , )

import matplotlib.pyplot as plt
import numpy as np

def plot_training_dataset(data, file_path):
    """
    Plots the training dataset with separate regions (boundaries and interior) and adds a legend.

    Parameters:
        data (dict): A dictionary containing boundary and interior datasets.
        file_path (str): Path to save the plot as a PDF.
    """
    # Extract data for all regions
    left_X, left_u = data["left"]
    right_X, right_u = data["right"]
    bottom_X, bottom_u = data["bottom"]
    top_X, top_u = data["top"]
    interior_X, interior_u = data["interior"]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each region with distinct markers and colors
    scatter_left = ax.scatter(left_X[:, 0], left_X[:, 1], c=left_u, s=20, cmap="viridis", label="Left Boundary", marker="o")
    scatter_right = ax.scatter(right_X[:, 0], right_X[:, 1], c=right_u, s=20, cmap="viridis", label="Right Boundary", marker="s")
    scatter_bottom = ax.scatter(bottom_X[:, 0], bottom_X[:, 1], c=bottom_u, s=20, cmap="viridis", label="Bottom Boundary", marker="^")
    scatter_top = ax.scatter(top_X[:, 0], top_X[:, 1], c=top_u, s=20, cmap="viridis", label="Top Boundary", marker="v")
    scatter_interior = ax.scatter(interior_X[:, 0], interior_X[:, 1], c=interior_u, s=10, cmap="viridis", label="Interior", marker="*")

    # Plot details
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title("Training Dataset", fontsize=14)
    ax.grid(True)

    # Add legend
    ax.legend(loc="upper right", fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(scatter_interior, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Values", fontsize=12)

    # Save the plot
    plt.tight_layout()
    plt.savefig(file_path, format="pdf", dpi=300)  # Save in PDF format with high resolution
    plt.close()  # Close the plot to free memory
