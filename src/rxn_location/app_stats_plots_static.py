"""
Static Statistics Plots Module

Generates static Matplotlib/Seaborn figures for the statistics mode of the GUI.
This is the static counterpart to app_stats_plots_interactive.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rxn_location import rc_stats_fncs as rcsf


def generate_static_plots(
    csv_file_path,
    dark_mode=False,
    selected_plots=None,
    x_var="b_imf_z",
    y_var="r_rc",
):
    """
    Generate static Matplotlib/Seaborn figures from a statistics CSV file.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file containing statistics data.
    dark_mode : bool
        If True, use a dark background template.
    selected_plots : list of str
        Which plot types to generate. Options: "Scatter Plots", "MMS Location Scatter Plot",
        "Box Plots", "Violin Plots".
    x_var : str
        Column name for the x-axis variable.
    y_var : str
        Column name for the y-axis variable.

    Returns
    -------
    dict
        A dictionary mapping plot titles to matplotlib Figure objects.
    """
    if selected_plots is None:
        selected_plots = ["Scatter Plots"]

    figures = {}

    if dark_mode:
        plt.style.use("dark_background")
        sns.set_context("notebook", font_scale=1.2)
        plt.rcParams.update({"axes.facecolor": "black", "figure.facecolor": "black"})
    else:
        plt.style.use("default")
        sns.set_context("notebook", font_scale=1.2)
        plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white"})

    var_labels = {
        "b_imf_z": "IMF Bz [nT]",
        "b_imf_x": "IMF Bx [nT]",
        "b_imf_y": "IMF By [nT]",
        "imf_clock_angle": "IMF Clock Angle [°]",
        "cone_angle": "Cone Angle [°]",
        "p_dyn": "Dynamic Pressure [nPa]",
        "msh_msp_shear": "Shear Angle [°]",
        "r_rc": "Reconnection Distance [Rₑ]",
        "delta_beta": "Δβ",
    }
    x_label = var_labels.get(x_var, x_var.replace("_", " ").title())
    y_label = var_labels.get(y_var, y_var.replace("_", " ").title())

    try:
        df = pd.read_csv(csv_file_path, index_col=False)
        df = rcsf.convert_wide_to_long(df)
    except Exception as e:
        return figures, f"Error reading CSV: {e}"

    model_names = ["Shear", "Reconnection Energy", "Exhaust Velocity", "Bisection"]
    model_filters = [
        ["shear", "Shear"],
        ["rx_en", "Reconnection Energy"],
        ["va_cs", "Exhaust Velocity"],
        ["bisection", "Bisection Field"],
    ]
    color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    df_list = []
    for filters in model_filters:
        df_model = df[df.method_used.isin(filters)].copy()
        if len(df_model) > 0:
            if "imf_clock_angle" in df_model.columns:
                df_model["imf_clock_angle"] = df_model["imf_clock_angle"].apply(
                    lambda x: 360 - x if x > 180 else x
                )
            if all(c in df_model.columns for c in ["b_imf_x", "b_imf_y", "b_imf_z"]):
                mag = np.sqrt(
                    df_model.b_imf_x**2 + df_model.b_imf_y**2 + df_model.b_imf_z**2
                )
                df_model["cone_angle"] = np.arccos(df_model.b_imf_x / mag) * 180 / np.pi
                df_model["bb"] = df_model.b_imf_y / mag
        df_list.append(df_model)

    def _setup_2x2_fig(title):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        axes = axes.flatten()
        return fig, axes

    if "Histograms" in selected_plots:
        try:
            fig, axes = _setup_2x2_fig("Reconnection Distance Histograms")
            for i, (df_n, name) in enumerate(zip(df_list, model_names)):
                ax = axes[i]
                if len(df_n) > 0 and "r_rc" in df_n.columns:
                    vals = df_n["r_rc"].dropna()
                    if len(vals) > 0:
                        sns.histplot(vals, ax=ax, color=color_list[i], bins=30)
                ax.set_title(name)
                ax.set_xlabel("Reconnection Distance [Rₑ]" if i >= 2 else "")
                ax.set_ylabel("Count" if i % 2 == 0 else "")
            fig.tight_layout()
            figures["Histograms"] = fig
        except Exception as e:
            print(f"Static Histogram error: {e}")

    if "KDE Plots" in selected_plots:
        try:
            fig, axes = _setup_2x2_fig(f"KDE Plots: {x_label}")
            for i, (df_n, name) in enumerate(zip(df_list, model_names)):
                ax = axes[i]
                if len(df_n) > 0 and x_var in df_n.columns:
                    vals = df_n[x_var].dropna()
                    if len(vals) > 0:
                        sns.kdeplot(vals, ax=ax, color=color_list[i], fill=True)
                ax.set_title(name)
                ax.set_xlabel(x_label if i >= 2 else "")
                ax.set_ylabel("Density" if i % 2 == 0 else "")
            fig.tight_layout()
            figures["KDE Plots"] = fig
        except Exception as e:
            print(f"Static KDE Plot error: {e}")

    if "2D Histograms" in selected_plots:
        try:
            colorscales = ["viridis", "cividis", "plasma", "magma"]
            fig, axes = _setup_2x2_fig(f"2D Histograms: {x_label} vs {y_label}")
            for i, (df_n, name) in enumerate(zip(df_list, model_names)):
                ax = axes[i]
                if len(df_n) > 0 and x_var in df_n.columns and y_var in df_n.columns:
                    sns.histplot(
                        data=df_n, x=x_var, y=y_var, bins=25, 
                        cbar=True, ax=ax, cmap=colorscales[i]
                    )
                ax.set_title(name)
                ax.set_xlabel(x_label if i >= 2 else "")
                ax.set_ylabel(y_label if i % 2 == 0 else "")
            fig.tight_layout()
            figures["2D Histograms"] = fig
        except Exception as e:
            print(f"Static 2D Histogram error: {e}")

    if "Scatter Plots" in selected_plots:
        try:
            fig, axes = _setup_2x2_fig(f"Scatter: {x_label} vs {y_label}")
            for i, (df_n, name) in enumerate(zip(df_list, model_names)):
                ax = axes[i]
                if len(df_n) > 0 and x_var in df_n.columns and y_var in df_n.columns:
                    ax.scatter(
                        df_n[x_var], df_n[y_var], 
                        c=color_list[i], alpha=0.7, edgecolors="white" if dark_mode else "black", linewidths=0.5
                    )
                ax.set_title(name)
                ax.set_xlabel(x_label if i >= 2 else "")
                ax.set_ylabel(y_label if i % 2 == 0 else "")
            fig.tight_layout()
            figures["Scatter Plots"] = fig
        except Exception as e:
            print(f"Static Scatter Plot error: {e}")

    if "MMS Location Scatter Plot" in selected_plots:
        try:
            pos_cols_xy = ["spc_pos_x", "spc_pos_y"]
            pos_cols_yz = ["spc_pos_y", "spc_pos_z"]
            alt_pos_cols_xy = ["x_gsm", "y_gsm"]
            alt_pos_cols_yz = ["y_gsm", "z_gsm"]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle("MMS Spacecraft Positions", fontsize=16)

            for i, (df_n, name) in enumerate(zip(df_list, model_names)):
                if len(df_n) == 0:
                    continue
                if all(c in df_n.columns for c in pos_cols_xy):
                    x_col, y_col = pos_cols_xy
                    y2_col, z_col = pos_cols_yz
                elif all(c in df_n.columns for c in alt_pos_cols_xy):
                    x_col, y_col = alt_pos_cols_xy
                    y2_col, z_col = alt_pos_cols_yz
                else:
                    continue

                ax1.scatter(df_n[x_col], df_n[y_col], label=name, color=color_list[i], alpha=0.7)
                ax2.scatter(df_n[y2_col], df_n[z_col], label=name, color=color_list[i], alpha=0.7)

            ax1.set_title("Spacecraft Y vs X")
            ax1.set_xlabel("MMS Position X [Rₑ]")
            ax1.set_ylabel("MMS Position Y [Rₑ]")
            ax1.legend()

            ax2.set_title("Spacecraft Z vs Y")
            ax2.set_xlabel("MMS Position Y [Rₑ]")
            ax2.set_ylabel("MMS Position Z [Rₑ]")
            fig.tight_layout()
            figures["MMS Location Scatter Plot"] = fig
        except Exception as e:
            print(f"Static MMS Location Plot error: {e}")
            
    # Do NOT run plt.close("all") here otherwise the caller receives destroyed figures
    return figures, None
