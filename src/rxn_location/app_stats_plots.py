import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rxn_location import seaborn_plots_fncs as spf
from rxn_location import rc_stats_fncs as rcsf

def generate_statistics_plots(csv_file_path, dark_mode=False, selected_plots=["Histograms"], x_var="b_imf_z", y_var="r_rc"):
    figures = {}
    
    # Label mappings
    var_options = {
        "b_imf_z": r"IMF $B_{\rm z}$ [nT]",
        "b_imf_x": r"IMF $B_{\rm x}$ [nT]",
        "b_imf_y": r"IMF $B_{\rm y}$ [nT]",
        "imf_clock_angle": r"IMF Clock Angle [$^{\circ}$]",
        "cone_angle": r"Cone Angle [$^{\circ}$]",
        "msh_msp_shear": r"Shear Angle [$^{\circ}$]",
        "r_rc": r"Reconnection Distance [$R_{\rm E}$]",
        "delta_beta": r"$\Delta \beta$"
    }
    
    x_label = var_options.get(x_var, x_var)
    y_label = var_options.get(y_var, y_var)
    
    # Read the data
    try:
        df = pd.read_csv(csv_file_path, index_col=False)
        df = rcsf.convert_wide_to_long(df)
    except Exception as e:
        return figures, f"Error reading CSV: {e}"

    if "Histograms" in selected_plots:
        try:
            fig, df_shear, df_rx_en, df_va_cs, df_bisec = rcsf.plot_hist(
                file_name=csv_file_path, 
                dark_mode=dark_mode, 
                return_fig=True,
                fig_folder="temp_figures", 
            )
            figures["Histograms"] = fig
        except Exception as e:
            print(f"Histogram error: {e}")

    # Prepare data for other plots
    if any(p in selected_plots for p in ["KDE Plots", "2D Histograms", "Scatter Plots", "MMS Location Scatter Plot"]):
        df_shear = df[df.method_used == "shear"].copy()
        df_rx_en = df[df.method_used == "rx_en"].copy()
        df_va_cs = df[df.method_used == "va_cs"].copy()
        df_bisec = df[df.method_used == "bisection"].copy()
        
        for df_n in [df_shear, df_rx_en, df_va_cs, df_bisec]:
            if len(df_n) > 0:
                if "imf_clock_angle" in df_n.columns:
                    df_n["imf_clock_angle"] = df_n["imf_clock_angle"].apply(lambda x: 360 - x if x > 180 else x)
                if all(col in df_n.columns for col in ["b_imf_x", "b_imf_y", "b_imf_z"]):
                    df_n["cone_angle"] = np.arccos(df_n.b_imf_x / np.sqrt(df_n.b_imf_x**2 + df_n.b_imf_y**2 + df_n.b_imf_z**2)) * 180 / np.pi
                    df_n["bb"] = df_n.b_imf_y / np.sqrt(df_n.b_imf_x**2 + df_n.b_imf_y**2 + df_n.b_imf_z**2)
        
        df_list = [df_shear, df_rx_en, df_va_cs, df_bisec]
        data_type = ["Shear", "Reconnection-Energy", "Exhaust-Velocity", "Bisection"]
        color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        if dark_mode:
            plt.style.use('dark_background')
            text_col, edge_col = "white", "black"
        else:
            plt.style.use('default')
            text_col, edge_col = "black", "white"
            
        font = {'family': 'serif', 'weight': 'normal', 'size': 10}
        plt.rc('font', **font)
        
        if "KDE Plots" in selected_plots:
            try:
                fig, axs = spf.seaborn_subplots(
                    df_list=df_list, 
                    keys=[x_var, y_var],
                    labels=[x_label, y_label],
                    x_lim=None, y_lim=None,
                    data_type=data_type, 
                    color_list=color_list, 
                    log_scale=False,
                    x_log_scale=False, 
                    y_log_scale=False,
                    fig_name="kde_temp", 
                    fig_format="pdf", 
                    nbins=[40, 40],
                    dark_mode=dark_mode, 
                    var_marker_size=True,
                    return_fig=True
                )
                figures["KDE Plots"] = fig
            except Exception as e:
                print(f"KDE Plot error: {e}")
                
        if "2D Histograms" in selected_plots:
            try:
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                cmap_list =  ["viridis", "cividis", "plasma", "magma"]
                
                # Get overall bounds
                all_x = np.concatenate([df_n[x_var].values for df_n in df_list if len(df_n) > 0 and x_var in df_n.columns])
                all_y = np.concatenate([df_n[y_var].values for df_n in df_list if len(df_n) > 0 and y_var in df_n.columns])
                
                if len(all_x) > 0 and len(all_y) > 0:
                    x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
                    y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
                    
                    if x_min == x_max: x_max = x_min + 1
                    if y_min == y_max: y_max = y_min + 1
                    
                    n_bins = 25
                    x_vals = np.linspace(x_min, x_max, n_bins)
                    y_vals = np.linspace(y_min, y_max, n_bins)
                    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
                    
                    x_step = x_vals[1] - x_vals[0] if len(x_vals) > 1 else 1
                    y_step = y_vals[1] - y_vals[0] if len(y_vals) > 1 else 1
                    
                    for i, df_n in enumerate(df_list):
                        if len(df_n) == 0 or x_var not in df_n.columns or y_var not in df_n.columns: 
                            continue
                            
                        z_vals = np.zeros((n_bins, n_bins))
                        
                        for ii, x in enumerate(x_vals):
                            for j, y in enumerate(y_vals):
                                subset = df_n.loc[(df_n[x_var] >= x) & (df_n[x_var] < x + x_step) & (df_n[y_var] >= y) & (df_n[y_var] < y + y_step)]
                                if "r_rc" in subset.columns:
                                    z_vals[j, ii] = np.nanmean(subset["r_rc"].values) if len(subset) > 0 else 0
                                else:
                                    z_vals[j, ii] = len(subset)
                                
                        ax = axs[i//2, i%2]
                        im = ax.pcolormesh(x_mesh, y_mesh, z_vals, cmap=cmap_list[i], shading='auto')
                        
                        if i < 2:
                            ax.set_xlabel("")
                            ax.set_xticklabels([])
                        else:
                            ax.set_xlabel(x_label, fontsize=15)
                            
                        if i%2 == 1:
                            ax.set_ylabel("")
                            ax.set_yticklabels([])
                        else:
                            ax.set_ylabel(y_label, fontsize=15)
                            
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
                        
                        ax.text(0.98, 0.02, data_type[i], fontsize=15, transform=ax.transAxes, va="bottom", ha="right", 
                                bbox=dict(facecolor="white" if not dark_mode else "black", alpha=1, edgecolor="black" if not dark_mode else "white", boxstyle='round,pad=0.2'))
                        
                        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.0)
                        cbar_label = r"Reconnection Distance $\left[R_{\rm E} \right]$" if "r_rc" in df_n.columns else "Count"
                        cbar.set_label(cbar_label, fontsize=0.75*15, labelpad=-28, y=0.755, rotation=90, va="top", ha="center", color=text_col)
                    
                    plt.subplots_adjust(hspace=0, wspace=0.1)
                    figures["2D Histograms"] = fig
            except Exception as e:
                print(f"2D Histograms error: {e}")
                
        if "Scatter Plots" in selected_plots:
            try:
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                
                # Get overall bounds
                valid_x_arrays = [df_n[x_var].values for df_n in df_list if len(df_n) > 0 and x_var in df_n.columns]
                valid_y_arrays = [df_n[y_var].values for df_n in df_list if len(df_n) > 0 and y_var in df_n.columns]
                
                if valid_x_arrays and valid_y_arrays:
                    all_x = np.concatenate(valid_x_arrays)
                    all_y = np.concatenate(valid_y_arrays)
                    x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
                    y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
                    
                    if x_min == x_max: x_max = x_min + 1
                    if y_min == y_max: y_max = y_min + 1
                    
                    x_pad = (x_max - x_min) * 0.05
                    y_pad = (y_max - y_min) * 0.05
                    x_min, x_max = x_min - x_pad, x_max + x_pad
                    y_min, y_max = y_min - y_pad, y_max + y_pad
                    
                    for i, df_n in enumerate(df_list):
                        if len(df_n) == 0 or x_var not in df_n.columns or y_var not in df_n.columns: 
                            continue
                            
                        ax = axs[i//2, i%2]
                        
                        sns.scatterplot(
                            x=x_var, y=y_var, 
                            data=df_n, 
                            ax=ax, 
                            alpha=0.6, 
                            color=color_list[i], 
                            edgecolor=edge_col
                        )
                        
                        if i < 2:
                            ax.set_xlabel("")
                            ax.set_xticklabels([])
                        else:
                            ax.set_xlabel(x_label, fontsize=15)
                            
                        if i%2 == 1:
                            ax.set_ylabel("")
                            ax.set_yticklabels([])
                        else:
                            ax.set_ylabel(y_label, fontsize=15)
                            
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
                        
                        ax.text(0.98, 0.02, data_type[i], fontsize=15, transform=ax.transAxes, va="bottom", ha="right", 
                                bbox=dict(facecolor="white" if not dark_mode else "black", alpha=1, edgecolor="black" if not dark_mode else "white", boxstyle='round,pad=0.2'))
                        
                        ax.grid(True, alpha=0.3, linestyle='--')
                    
                    plt.subplots_adjust(hspace=0.05, wspace=0.05)
                    figures["Scatter Plots"] = fig
            except Exception as e:
                print(f"Scatter Plot error: {e}")
                
        if "MMS Location Scatter Plot" in selected_plots:
            try:
                fig, axs = plt.subplots(1, 2, figsize=(14, 6))
                
                for i, df_n in enumerate(df_list):
                    if len(df_n) == 0: continue
                    if "spc_pos_x" in df_n.columns and "spc_pos_y" in df_n.columns and "spc_pos_z" in df_n.columns:
                        axs[0].scatter(df_n["spc_pos_x"], df_n["spc_pos_y"], label=data_type[i], color=color_list[i], alpha=0.7)
                        axs[1].scatter(df_n["spc_pos_y"], df_n["spc_pos_z"], label=data_type[i], color=color_list[i], alpha=0.7)
                        
                axs[0].set_xlabel(r"MMS Position $X$ [$R_{\rm E}$]", fontsize=12)
                axs[0].set_ylabel(r"MMS Position $Y$ [$R_{\rm E}$]", fontsize=12)
                axs[0].set_title(r"Spacecraft $Y$ vs $X$", fontsize=14)
                axs[0].grid(True, linestyle="--", alpha=0.5)
                axs[0].legend()
                
                axs[1].set_xlabel(r"MMS Position $Y$ [$R_{\rm E}$]", fontsize=12)
                axs[1].set_ylabel(r"MMS Position $Z$ [$R_{\rm E}$]", fontsize=12)
                axs[1].set_title(r"Spacecraft $Z$ vs $Y$", fontsize=14)
                axs[1].grid(True, linestyle="--", alpha=0.5)
                axs[1].legend()
                
                plt.tight_layout()
                figures["MMS Location Scatter Plot"] = fig
                
            except Exception as e:
                print(f"MMS Location Plot error: {e}")

    return figures, None
