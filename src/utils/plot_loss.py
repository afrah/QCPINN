import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from src.utils.color import model_color
from src.utils.utilities import PLOT_STYLES
from typing import Dict, Tuple, List, Any


def exponential_moving_average(data, alpha=0.1):
    """Applies Exponential Moving Average (EMA) smoothing to the data."""
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def smooth_loss(data, alpha=0.1, window_length=51, polyorder=3):
    """Applies a combination of EMA and Savitzky-Golay smoothing to the data."""
    ema_data = exponential_moving_average(data, alpha=alpha)
    if len(ema_data) < window_length:  # Avoid issues with short data
        window_length = len(ema_data) - 1 if len(ema_data) % 2 == 0 else len(ema_data)
    smoothed_data = savgol_filter(
        ema_data, window_length=window_length, polyorder=polyorder
    )
    return smoothed_data


def spline_smoothing(data, s=0.5):
    """Applies spline smoothing to the data."""
    x = np.arange(len(data))
    spline = UnivariateSpline(x, data, s=s)
    return spline(x)


def smoothed_min(data):
    # Calculate rolling minimum
    window_size = 1000
    min_values = np.minimum.accumulate(data)
    for i in range(window_size, len(data)):
        min_values[i] = min(data[i - window_size + 1 : i + 1])

    # Smooth the minimum values
    window_length = 51  # must be odd and less than data size
    poly_order = 3  # must be less than window_length
    smoothed_min = savgol_filter(min_values, window_length, poly_order)
    return smoothed_min



def plot_loss_history(all_loss_history , save_path=None, y_max=None , legend=False):
    """
    Usage:
    plot_loss_history(data_list, save_path=None, y_max=None)
    """
    # Set font sizes
    SMALL_SIZE = 10    # For ticks
    MEDIUM_SIZE = 14   # For labels
    BIGGER_SIZE = 16   # For title
    LEGEND_SIZE = 12   # For legend

    # Set ggplot style
    plt.style.use("ggplot")

    fig, ax = plt.subplots()
    fig.set_size_inches([5, 5])


    def create_plot_config(all_loss_history: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Create plot configuration for visualization"""
        try:
            return [
                {
                    "data": all_loss_history[model_name],
                    "color": model_color[model_name],
                    "name": model_name.upper() if model_name in ["cv", "gcv"] else model_name,
                    "alpha": 1.0,
                    "window": 100,
                    "show_avg": False,
                    "show_lower": False,
                    "linestyle": PLOT_STYLES[model_name],
                    "linewidth": 2,
                    "label": model_name.upper() if model_name in ["cv", "gcv"] else model_name  # Explicit label
                }
                for model_name in all_loss_history.keys()
            ]
        except Exception as e:
            print(f"Error creating plot configuration: {e}")
            raise

    data_list = create_plot_config(all_loss_history)

    # Print loss values at iteration 12500
    print("\nLoss values at iteration 12500:")
    print("-" * 40)
    
    for entry in data_list:
        data = entry["data"]
        color = entry["color"]
        name = entry["name"]
        alpha = entry["alpha"]
        window = entry["window"]
        show_avg = entry["show_avg"]
        show_lower = entry["show_lower"]
        linestyle = entry["linestyle"]
        linewidth = entry["linewidth"]
        # Apply smoothing and plot the smoothed data
        smooth_alpha = 0.1
        polyorder = 1
        smoothed_data = smooth_loss(data, smooth_alpha, window, polyorder)

        
        # Print loss value at iteration 12500
        final_loss = smoothed_data[-1]
        print(f"{name:<20}: {final_loss:.6f}")
        # Plot the original data
        sns.lineplot(
            x=np.arange(len(data)),
            y=smoothed_data,
            ax=ax,
            label=f"{name}",
            color=color,
            linestyle=linestyle,
            alpha=alpha,
            linewidth=linewidth
        )

        if show_avg:
            sns.lineplot(
                x=np.arange(len(smoothed_data)),
                y=smoothed_data,
                ax=ax,
                color=color,
                linewidth=0.5,
            )
        if show_lower:
            smoothed_lower = smoothed_min(data)
            sns.lineplot(
                x=np.arange(len(smoothed_lower)),
                y=smoothed_lower,
                ax=ax,
                color=color,
                linewidth=1.5,
            )

    print("-" * 40)

    ax.set_yscale("log")

    # # Set x-axis ticks every 5000 epochs
    x_ticks = np.arange(0, len(data),  int(len(data)/4))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)

    # Increase font sizes for labels and ticks
    ax.set_xlabel("Epochs →", fontsize=MEDIUM_SIZE, color="grey", labelpad=10)
    ax.set_ylabel("Loss (log) →", fontsize=MEDIUM_SIZE, color="grey", labelpad=10)
    ax.tick_params(axis="both", labelsize=SMALL_SIZE, colors="grey", pad=8)
    
    # Style spines
    ax.spines["top"].set_color("grey")
    ax.spines["bottom"].set_color("grey")
    ax.spines["left"].set_color("grey")
    ax.spines["right"].set_color("grey")
    
    # Increase spine width for better visibility
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Set the background color to white and grid lines to very light grey
    ax.set_facecolor("white")
    ax.grid(True, color="lightgrey", linewidth=1.2)

    if y_max is not None:
        ax.set_ylim(top=y_max)



    # Handle legend
    existing_legend = ax.get_legend()
    if existing_legend is not None:
        existing_legend.remove()

    if legend:
        # Make the legend more readable with increased font size
        legend = ax.legend(
            loc='upper right',
            fontsize=LEGEND_SIZE,
            frameon=True,
            ncol=1,  # Changed to 3 columns
            borderaxespad=0.5,
            bbox_to_anchor=(1, 1),
            handlelength=1.5,
            handletextpad=0.5
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.5)
    plt.tight_layout()

    
    # Save with high DPI for better quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close('all',)

def plot_cv_losses(iterations, loss_r_values, loss_bc_values, file_name, fig_flag=None):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
    
    # Set font sizes
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    
    # Update font sizes
    plt.rc('font', size=SMALL_SIZE)          
    plt.rc('axes', titlesize=BIGGER_SIZE)    
    plt.rc('axes', labelsize=MEDIUM_SIZE)    
    plt.rc('xtick', labelsize=SMALL_SIZE)    
    plt.rc('ytick', labelsize=SMALL_SIZE)    
    plt.rc('legend', fontsize=MEDIUM_SIZE)   
    
    # Apply smoothing to get trend lines
    window_length = min(51, len(loss_r_values) - 1 if len(loss_r_values) % 2 == 0 else len(loss_r_values))
    smooth_r = smooth_loss(loss_r_values, alpha=0.1, window_length=window_length, polyorder=2)
    smooth_bc = smooth_loss(loss_bc_values, alpha=0.1, window_length=window_length, polyorder=2)
    
    # Plot loss_r
    ax1.plot(iterations, loss_r_values, 'b-', label='loss_r', linewidth=1)
    ax1.plot(iterations, smooth_r, 'b--',  linewidth=1.0, alpha=0.5)
    ax1.set_ylabel('Loss_r', fontsize=MEDIUM_SIZE)
    if fig_flag == "helmholtz":
        ax1.set_ylim(bottom=0, top=max(loss_r_values)+1000)
    # else:
    #     ax1.set_ylim(bottom=0, top=max(loss_r_values)+0.0001)
    # Set only two y-ticks for loss_r
    y_min, y_max = ax1.get_ylim()
    ax1.set_yticks([y_min, y_max])
    ax1.set_yticklabels([f"{y_min:.1e}", f"{y_max:.1e}"])
    
    ax1.grid(True)
    ax1.legend(prop={'size': MEDIUM_SIZE})
    ax1.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
    # Hide x-axis labels and ticks for top plot
    ax1.set_xticklabels([])
    
    # Plot loss_bc
    ax2.plot(iterations, loss_bc_values, 'r-', label='loss_bc', linewidth=1)
    ax2.plot(iterations, smooth_bc, 'r--',  linewidth=1.0 , alpha=0.5)
    if fig_flag == "helmholtz":
        ax2.set_ylim(bottom=0, top=max(loss_bc_values)+0.1)
    # else:
    #     ax2.set_ylim(bottom=0, top=max(loss_bc_values)+0.001)
    # Set only two y-ticks for loss_bc
    y_min, y_max = ax2.get_ylim()
    ax2.set_yticks([y_min, y_max])
    ax2.set_yticklabels([f"{y_min:.1e}", f"{y_max:.1e}"])
    
    ax2.set_xlabel('Iteration', fontsize=MEDIUM_SIZE)
    ax2.set_ylabel('Loss_bc', fontsize=MEDIUM_SIZE)
    ax2.grid(True)
    ax2.legend(prop={'size': MEDIUM_SIZE})
    ax2.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot as PDF with 300 DPI
    plt.savefig(file_name, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close('all')