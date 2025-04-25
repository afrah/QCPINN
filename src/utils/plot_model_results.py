import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from src.utils.cmap import orange_cmap2

SOLUTION_MAP  = "rainbow"
ERROR_MAP = orange_cmap2

def plt_model_results(logger, X_star, u_star, f_star, results, problem=None):
    if problem == "wave":
        data = {
            "u": {
                "exact": u_star,
                "classical_prediction": results["classical"],
                "classical_error": np.abs(u_star - results["classical"]),
                "dv_prediction": results["angle_cascade"],
                "dv_error": np.abs(u_star - results["angle_cascade"]),
                "title": r"$u(x)$",
                "pred_title": r"$\hat{u}(x)$",
            },
        }
        
        # Calculate global min/max for solution and error plots
        value = data["u"]
        solution_data = [value["exact"], value["classical_prediction"], value["dv_prediction"]]
        error_data = [value["classical_error"], value["dv_error"]]
        
        solution_min = min(np.min(arr) for arr in solution_data)
        solution_max = max(np.max(arr) for arr in solution_data)
        error_min = min(np.min(arr) for arr in error_data)
        error_max = max(np.max(arr) for arr in error_data)
        
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        plt.style.use('default')
        fig.patch.set_facecolor('white')
        
        column_titles = [
            "Exact solution $u(x)$",
            "PINN prediction $\\hat{u}(x)$", 
            "PINN error",
            "QCPINN Prediction $\\hat{u}(x)$",
            "QCPINN eror",
        ]
        
        for col in range(5):
            ax = axs[col]
            ax.set_facecolor('white')
            ax.set_aspect('equal', adjustable='box')
            
            # Determine plot data and settings based on column
            if col == 0:
                plot_data = value["exact"]
                cmap = SOLUTION_MAP
                vmin, vmax = solution_min, solution_max
            elif col == 1:
                plot_data = value["classical_prediction"]
                cmap = SOLUTION_MAP
                vmin, vmax = solution_min, solution_max
            elif col == 2:
                plot_data = value["classical_error"]
                cmap = orange_cmap2
                vmin, vmax = 0.0 , error_max
            elif col == 3:
                plot_data = value["dv_prediction"]
                cmap = SOLUTION_MAP
                vmin, vmax = solution_min, solution_max
            else:  # col == 4
                plot_data = value["dv_error"]
                cmap = orange_cmap2
                vmin, vmax =  0.0, error_max
            
            x_fine = np.linspace(X_star[:, 0].min(), X_star[:, 0].max(), 100)
            y_fine = np.linspace(X_star[:, 1].min(), X_star[:, 1].max(), 100)
            X, Y = np.meshgrid(x_fine, y_fine)
            
            points = np.c_[X_star[:, 0], X_star[:, 1]]
            values = plot_data.flatten()
            levels_value = 100
            levels = np.linspace(vmin, vmax, levels_value)
            Z = griddata(points, values, (X, Y), method='cubic')
            
            mesh = ax.contourf(X, Y, Z,
                             cmap=cmap,
                             vmin=vmin,
                             vmax=vmax,
                             levels=levels)
            
            ax.contour(X, Y, Z, levels=levels,
                      colors='white', linewidths=0.5, alpha=0.3)
            
            ax.set_xlim(X_star[:, 0].min(), X_star[:, 0].max())
            ax.set_ylim(X_star[:, 1].min(), X_star[:, 1].max())
            
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ticks = np.linspace(vmin, vmax, 3)
            cb = fig.colorbar(mesh, cax=cax, format='%.1e', ticks=ticks)
            cb.ax.tick_params(labelsize=13)
            
            if col == 0:
                x_ticks = np.linspace(X_star[:, 0].min(), X_star[:, 0].max(), 3)
                y_ticks = np.linspace(X_star[:, 1].min(), X_star[:, 1].max(), 4)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ax.set_xticklabels([f'{x:.1f}' for x in x_ticks])
                ax.set_yticklabels([f'{y:.1f}' for y in y_ticks])
                ax.set_xlabel(r"$x_1$→", fontsize=17)
                ax.set_ylabel(r"$x_2$→", fontsize=17)
                ax.tick_params(axis='both', which='major', labelsize=15)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            
            if col == 0:
                subtitle = value['title']
            elif col in [1, 3]:
                subtitle = value['pred_title']
            else:
                subtitle = ""
            title = f"{column_titles[col]}"
            ax.set_title(title, pad=10, fontsize=14)

        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            bottom=0.15,
            top=0.92,
            wspace=0.5,
        )
        
        path = os.path.join(logger.get_output_dir(), "prediction.png")
        plt.savefig(path, 
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none',
                    pad_inches=0.1)
        plt.show()
        plt.close()

    else:
        data = {
            "u": {
                "exact": u_star,
                "classical_prediction": results["classical"][0],
                "classical_error": np.abs(u_star - results["classical"][0]),
                "dv_prediction": results["angle_cascade"][0],
                "dv_error": np.abs(u_star - results["angle_cascade"][0]),
                "title": r"$u(x)$",
                "pred_title": r"$\hat{u}(x)$",
            },
            "f": {
                "exact": f_star,
                "classical_prediction": results["classical"][1],
                "classical_error": np.abs(f_star - results["classical"][1]),
                "dv_prediction": results["angle_cascade"][1],
                "dv_error": np.abs(f_star - results["angle_cascade"][1]),
                "title": r"$f(x)$",
                "pred_title": r"$\hat{f}(x)$",
            },
        }
        
        # Calculate global min/max for each row's solution and error plots
        solution_mins = []
        solution_maxs = []
        error_mins = []
        error_maxs = []
        
        for value in data.values():
            solution_data = [value["exact"], value["classical_prediction"], value["dv_prediction"]]
            error_data = [value["classical_error"], value["dv_error"]]
            
            solution_mins.append(min(np.min(arr) for arr in solution_data))
            solution_maxs.append(max(np.max(arr) for arr in solution_data))
            error_mins.append(min(np.min(arr) for arr in error_data))
            error_maxs.append(max(np.max(arr) for arr in error_data))
        
        fig, axs = plt.subplots(2, 5, figsize=(20, 8))
        plt.style.use('default')
        fig.patch.set_facecolor('white')
        column_titles = [
            "Exact solution $u(x)$",
            "PINN prediction $\\hat{u}(x)$", 
            "PINN error",
            "QCPINN prediction $\\hat{u}(x)$",
            "QCPINN error",  
            "Exact solution $f(x)$",
            "PINN prediction $\\hat{f}(x)$", 
            "PINN error",
            "QCPINN prediction $\\hat{f}(x)$",
            "QCPINN error",  
        ]
        
        for row, (key, value) in enumerate(data.items()):
            solution_min, solution_max = solution_mins[row], solution_maxs[row]
            error_min, error_max = error_mins[row], error_maxs[row]
            
            for col in range(5):
                subplot_idx = row * 5 + col
                ax = axs[row, col]
                
                ax.set_facecolor('white')
                ax.set_aspect('equal', adjustable='box')
                
                # Determine plot data and settings based on column
                if col == 0:
                    plot_data = value["exact"]
                    cmap = SOLUTION_MAP
                    vmin, vmax = solution_min, solution_max
                elif col == 1:
                    plot_data = value["classical_prediction"]
                    cmap = SOLUTION_MAP
                    vmin, vmax = solution_min, solution_max
                elif col == 2:
                    plot_data = value["classical_error"]
                    cmap = orange_cmap2
                    vmin, vmax =  0.0, error_max
                elif col == 3:
                    plot_data = value["dv_prediction"]
                    cmap = SOLUTION_MAP
                    vmin, vmax = solution_min, solution_max
                else:  # col == 4
                    plot_data = value["dv_error"]
                    cmap = orange_cmap2
                    vmin, vmax =  0.0, error_max
                
                x_fine = np.linspace(X_star[:, 0].min(), X_star[:, 0].max(), 100)
                y_fine = np.linspace(X_star[:, 1].min(), X_star[:, 1].max(), 100)
                X, Y = np.meshgrid(x_fine, y_fine)
                
                points = np.c_[X_star[:, 0], X_star[:, 1]]
                values = plot_data.flatten()
                levels_value = 100
                levels = np.linspace(vmin, vmax, levels_value)
                Z = griddata(points, values, (X, Y), method='cubic')
                
                mesh = ax.contourf(X, Y, Z,
                                 cmap=cmap,
                                 vmin=vmin,
                                 vmax=vmax,
                                 levels=levels)
                
                ax.contour(X, Y, Z, levels=levels,
                          colors='white', linewidths=0.5, alpha=0.3)
                
                ax.set_xlim(X_star[:, 0].min(), X_star[:, 0].max())
                ax.set_ylim(X_star[:, 1].min(), X_star[:, 1].max())
                
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.5)
                
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                ticks = np.linspace(vmin, vmax, 3)
                cb = fig.colorbar(mesh, cax=cax, format='%.1e', ticks=ticks)
                cb.ax.tick_params(labelsize=13)
                
                if row == 1 and col == 0:
                    x_ticks = np.linspace(X_star[:, 0].min(), X_star[:, 0].max(), 3)
                    y_ticks = np.linspace(X_star[:, 1].min(), X_star[:, 1].max(), 4)
                    ax.set_xticks(x_ticks)
                    ax.set_yticks(y_ticks)
                    ax.set_xticklabels([f'{x:.1f}' for x in x_ticks])
                    ax.set_yticklabels([f'{y:.1f}' for y in y_ticks])
                    ax.set_xlabel(r"$x_1$→", fontsize=17)
                    ax.set_ylabel(r"$x_2$→", fontsize=17)
                    ax.tick_params(axis='both', which='major', labelsize=15)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                if col == 0:
                    subtitle = value['title']
                elif col in [1, 3]:
                    subtitle = value['pred_title']
                else:
                    subtitle = ""
                title = f"{column_titles[subplot_idx]}"
                ax.set_title(title, pad=10, fontsize=14)

        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            bottom=0.15,
            top=0.92,
            wspace=0.5,
            hspace=0.4,
        )
        
        path = os.path.join(logger.get_output_dir(), "prediction.png")
        plt.savefig(path, 
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none',
                    pad_inches=0.1)
        plt.show()
        plt.close()