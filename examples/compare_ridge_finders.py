from pandas.core import frame
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import RegularGridInterpolator
from rxn_location import rx_model_funcs as rmf
from improved_ridge_finder import trace_bisection_xline
from improved_ridge_finder import find_xline_radial_bisection
from skimage.filters import frangi
import scipy.ndimage as ndimage
from importlib import reload

reload(rmf)
# reload(trace_bisection_xline)

import os

def main():
    # Load inputs from the pickle file generated previously by rx_code.py, if available
    # Or run the model directly. Let's just run the model directly for a specific time.
    
    mms_probe_num = "3"
    min_max_val = 20
    dr = 0.25
    y_min = -min_max_val
    y_max = min_max_val
    z_min = -min_max_val
    z_max = min_max_val
    model_type = "t96"
    trange = ["2016-10-02 20:02:30"]
    
    USE_FRANGI_FOR_START = False
    SMOOTHING_POINTS_ORIGINAL = 15
    SMOOTHING_POINTS_BISECTION = 5
    FIELD_SMOOTHING_SIGMA = 5  # Sigma for gaussian filter on the plotted background field
    
    model_inputs = {
        "trange": trange,
        "probe": None,
        "omni_level": "hro",
        "mms_probe_num": mms_probe_num,
        "model_type": model_type,
        "m_p": 1,
        "dr": dr,
        "min_max_val": min_max_val,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
        "save_data": False,
        "nprocesses": None,
    }

    # ==========================
    # DATA CACHING FLAG
    # ==========================
    # Set this to True to skip running the physics model and just reload the previous data for faster plotting tweaks.
    # Note: If you change any rx_model inputs above (like trange), you MUST set this to False to recompute!
    figure_only_mode = True
    from pathlib import Path
    data_cache_file = Path("compare_ridge_finders_data.pkl")

    if figure_only_mode and data_cache_file.exists():
        import pickle
        print(f"Loading cached model data from {data_cache_file}...")
        with open(data_cache_file, "rb") as f:
            res = pickle.load(f)
    else:
        print("Running physics model (rx_model)...")
        res = rmf.rx_model(**model_inputs)
        import pickle
        with open(data_cache_file, "wb") as f:
            pickle.dump(res, f)
        print(f"Saved model data to {data_cache_file}")

    bx, by, bz, shear, rx_en, va_cs, bisec_msp, bisec_msh, sw_params, x_shu, y_shu, z_shu, b_msx, b_msy, b_msz = res
    
    shear_norm = (shear - np.nanmin(shear)) / (np.nanmax(shear) - np.nanmin(shear))
    rx_en_norm = (rx_en - np.nanmin(rx_en)) / (np.nanmax(rx_en) - np.nanmin(rx_en))
    va_cs_norm = (va_cs - np.nanmin(va_cs)) / (np.nanmax(va_cs) - np.nanmin(va_cs))
    bisec_msp_norm = (bisec_msp - np.nanmin(bisec_msp)) / (np.nanmax(bisec_msp) - np.nanmin(bisec_msp))
    
    # Smooth all fields before running any algorithms
    if FIELD_SMOOTHING_SIGMA > 0:
        sigma_xy = [FIELD_SMOOTHING_SIGMA, FIELD_SMOOTHING_SIGMA]
        shear_norm = ndimage.gaussian_filter(shear_norm, sigma=sigma_xy, mode='reflect')
        rx_en_norm = ndimage.gaussian_filter(rx_en_norm, sigma=sigma_xy, mode='reflect')
        va_cs_norm = ndimage.gaussian_filter(va_cs_norm, sigma=sigma_xy, mode='reflect')
        bisec_msp_norm = ndimage.gaussian_filter(bisec_msp_norm, sigma=sigma_xy, mode='reflect')
        
        bx = ndimage.gaussian_filter(bx, sigma=sigma_xy, mode='reflect')
        by = ndimage.gaussian_filter(by, sigma=sigma_xy, mode='reflect')
        bz = ndimage.gaussian_filter(bz, sigma=sigma_xy, mode='reflect')
        b_msx = ndimage.gaussian_filter(b_msx, sigma=sigma_xy, mode='reflect')
        b_msy = ndimage.gaussian_filter(b_msy, sigma=sigma_xy, mode='reflect')
        b_msz = ndimage.gaussian_filter(b_msz, sigma=sigma_xy, mode='reflect')

    # Run the original ridge_finder_multiple
    figure_inputs = {
        "image": [shear_norm, rx_en_norm, va_cs_norm, bisec_msp_norm],
        "convolution_order": [1, 1, 1, 1],
        "t_range": trange,
        "b_imf": np.round(sw_params["b_imf"], 2),
        "b_msh": np.round(sw_params["mms_b_gsm"], 2),
        "xrange": [y_min, y_max],
        "yrange": [z_min, z_max],
        "mms_probe_num": mms_probe_num,
        "mms_sc_pos": np.round(sw_params["mms_sc_pos"], 2),
        "dr": dr,
        "dipole_tilt_angle": sw_params["ps"],
        "p_dyn": np.round(sw_params["p_dyn"], 2),
        "imf_clock_angle": sw_params["imf_clock_angle"],
        "np_imf": np.round(sw_params["np"], 2),
        "v_imf": np.round(sw_params["v_imf"], 2),
        "sym_h": np.round(sw_params["sym_h"], 2),
        "sigma": [2, 2, 2, 2],
        "mode": "nearest",
        "alpha": 1,
        "vmin": [0, 0, 0, 0],
        "vmax": [1, 1, 1, 1],
        "cmap_list": ["viridis", "viridis", "viridis", "viridis"],
        "draw_patch": [True, True, True, True],
        "draw_ridge": [True, True, True, True],
        "save_fig": True,
        "fig_name": "new",
        "fig_format": "png",
        "c_label": ["Shear", "Reconnection Energy", "Exhaust Velocity", "Bisection Field"],
        "wspace": 0.0,
        "hspace": 0.15,
        "fig_size": (8.775, 10),
        "box_style": dict(boxstyle="round", color="w", alpha=0.8),
        "title_y_pos": 1.09,
        "interpolation": "nearest",
        "tsy_model": model_type,
        "dark_mode": True,
        "save_rc_file": False,
        "fig_version": "v001",
        "plot_orig_xline": True,
        "plot_bisec_xline": False,
        "plot_ivp_xline": True,
    }
    
    print("Running old algorithm (ridge_finder_multiple)...")
    filtered_inputs = {k: v for k, v in figure_inputs.items() if not k.startswith("plot_")}
    y_vals, x_intr_vals_list, y_intr_vals_list, _ = rmf.ridge_finder_multiple(**filtered_inputs)
    
    # Run the new bisection algorithm
    # Interpolator needs strictly monotonically increasing axes.
    print("Running new algorithms...")
    
    # Create the 1D axes arrays
    y_1d = np.linspace(y_min, y_max, shear_norm.shape[0])
    z_1d = np.linspace(z_min, z_max, shear_norm.shape[1])
    
    # RegularGridInterpolator requires strictly ascending arrays
    # If linspace starts from y_max to y_min, reverse it
    if y_1d[0] > y_1d[-1]:
        y_1d = y_1d[::-1]
        shear_norm = shear_norm[::-1, :]
        rx_en_norm = rx_en_norm[::-1, :]
        va_cs_norm = va_cs_norm[::-1, :]
        bisec_msp_norm = bisec_msp_norm[::-1, :]
        bx = bx[::-1, :]; by = by[::-1, :]; bz = bz[::-1, :]
        b_msx = b_msx[::-1, :]; b_msy = b_msy[::-1, :]; b_msz = b_msz[::-1, :]
        
    if z_1d[0] > z_1d[-1]:
        z_1d = z_1d[::-1]
        shear_norm = shear_norm[:, ::-1]
        rx_en_norm = rx_en_norm[:, ::-1]
        va_cs_norm = va_cs_norm[:, ::-1]
        bisec_msp_norm = bisec_msp_norm[:, ::-1]
        bx = bx[:, ::-1]; by = by[:, ::-1]; bz = bz[:, ::-1]
        b_msx = b_msx[:, ::-1]; b_msy = b_msy[:, ::-1]; b_msz = b_msz[:, ::-1]
        
    if FIELD_SMOOTHING_SIGMA > 0:
        sigma_xy = [FIELD_SMOOTHING_SIGMA, FIELD_SMOOTHING_SIGMA]
        bx = ndimage.gaussian_filter(bx, sigma=sigma_xy, mode='reflect')
        by = ndimage.gaussian_filter(by, sigma=sigma_xy, mode='reflect')
        bz = ndimage.gaussian_filter(bz, sigma=sigma_xy, mode='reflect')
        b_msx = ndimage.gaussian_filter(b_msx, sigma=sigma_xy, mode='reflect')
        b_msy = ndimage.gaussian_filter(b_msy, sigma=sigma_xy, mode='reflect')
        b_msz = ndimage.gaussian_filter(b_msz, sigma=sigma_xy, mode='reflect')
        
    fields_dict = {
        "shear": (0, shear_norm, "Shear"),
        "rx_en": (1, rx_en_norm, "Reconnection Energy"),
        "va_cs": (2, va_cs_norm, "Exhaust (Va/Cs)"),
        "bisec_msp": (3, bisec_msp_norm, "Bisection (MSP)")
    }
    
    # For the IVP algorithm, we need interpolators for the magnetic fields
    # b_msh and b_msp.
    b_msp_interp_x = RegularGridInterpolator((y_1d, z_1d), bx, bounds_error=False, fill_value=np.nan)
    b_msp_interp_y = RegularGridInterpolator((y_1d, z_1d), by, bounds_error=False, fill_value=np.nan)
    b_msp_interp_z = RegularGridInterpolator((y_1d, z_1d), bz, bounds_error=False, fill_value=np.nan)
    
    b_msh_interp_x = RegularGridInterpolator((y_1d, z_1d), b_msx, bounds_error=False, fill_value=np.nan)
    b_msh_interp_y = RegularGridInterpolator((y_1d, z_1d), b_msy, bounds_error=False, fill_value=np.nan)
    b_msh_interp_z = RegularGridInterpolator((y_1d, z_1d), b_msz, bounds_error=False, fill_value=np.nan)
    
    def get_b_msp(y, z):
        res = [b_msp_interp_x([y, z])[0], b_msp_interp_y([y, z])[0], b_msp_interp_z([y, z])[0]]
        return np.array(res)
        
    def get_b_msh(y, z):
        res = [b_msh_interp_x([y, z])[0], b_msh_interp_y([y, z])[0], b_msh_interp_z([y, z])[0]]
        return np.array(res)
        
    Y_grid, Z_grid = np.meshgrid(y_1d, z_1d, indexing='ij')
    
    print("Plotting comparison...")

    if figure_inputs["dark_mode"]:
        print("Dark mode")
        plt.style.use("dark_background")
        mtick_color = "w"  # color of the minor tick lines
        label_color = "w"  # color of the tick labels
        clabel_color = "w"  # color of the colorbar label
    else:
        print("Light mode")
        plt.style.use("default")
        # tick_color = "k"  # color of the tick lines
        mtick_color = "k"  # color of the minor tick lines
        label_color = "k"  # color of the tick labels
        clabel_color = "k"  # color of the colorbar label

    font = {"family": "serif", "weight": "normal", "size": 10}
    plt.rc("font", **font)
    plt.rc("text", usetex=True)

    fig = plt.figure(num=None, figsize=figure_inputs["fig_size"], dpi=200)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=figure_inputs["wspace"], hspace=figure_inputs["hspace"])
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])

    # Set the font size for the axes
    label_size = 20  # fontsize for x and y labels
    t_label_size = 18  # fontsize for tick label
    c_label_size = 18  # fontsize for colorbar label
    ct_tick_size = 14  # fontsize for colorbar tick labels
    l_label_size = 14  # fontsize for legend label

    tick_len = 10  # length of the tick lines
    mtick_len = 7  # length of the minor tick lines
    tick_width = 1  # tick width in points
    mtick_width = 0.7  # minor tick width in points

    # box_style = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    if figure_inputs["dark_mode"]:
        box_style = dict(boxstyle="round", color="w", alpha=0.8, linewidth=1)
    else:
        box_style = dict(boxstyle="round", color="w", alpha=0.8, linewidth=1)

    for key, (f_idx, target_field, target_name) in fields_dict.items():
        if USE_FRANGI_FOR_START:
            # Apply Gaussian filter and Frangi ridge filter to the chosen field
            # and remove the broad plateaus and noise.
            field_smooth = ndimage.gaussian_filter(target_field, sigma=[5, 5], mode='reflect')
            kwargs = {"sigmas": [2.5, 3], "mode": "reflect", "black_ridges": False}
            field_frangi = frangi(field_smooth, **kwargs)
            max_field = field_frangi
        else:
            max_field = target_field
            
        # Find the starting point (global maximum of the chosen field on the grid)
        max_idx = np.unravel_index(np.nanargmax(max_field), max_field.shape)
        y_start = y_1d[max_idx[0]]
        z_start = z_1d[max_idx[1]]
        print(f"[{target_name}] Starting IVP trace at local maximum: y={y_start:.2f}, z={z_start:.2f}")
        y_ivp, z_ivp = trace_bisection_xline(
            y_start, z_start, get_b_msh, get_b_msp, 
            step_size=0.25, max_steps=1000, bounds=min_max_val,
            enforce_monotonic_y=False
        )
        print(f"[{target_name}] IVP array: len={len(y_ivp)}, max_Y={np.max(y_ivp):.2f}, min_Y={np.min(y_ivp):.2f}, max_Z={np.max(z_ivp):.2f}, min_Z={np.min(z_ivp):.2f}")
        # Radial Bisection
        target_interp = RegularGridInterpolator((y_1d, z_1d), target_field, bounds_error=False, fill_value=np.nan)
        eval_func = lambda y, z: target_interp([y, z])[0]
        y_rad, z_rad = find_xline_radial_bisection(eval_func, r_bounds=(0.1, min_max_val * np.sqrt(2)), theta_steps=180)
        
        # Plotting
        import matplotlib.patches as patches
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        ax = plt.subplot(gs[f_idx // 2, f_idx % 2])
        cmaps = ["viridis", "cividis", "plasma", "magma"]
        cmap = cmaps[f_idx] if f_idx < len(cmaps) else "viridis"
        
        # target_field is already smoothed from earlier
        im = ax.pcolormesh(Y_grid, Z_grid, target_field, shading='auto', cmap=cmap)
        ax.set_aspect("equal")
        divider1 = make_axes_locatable(ax)
        
        if figure_inputs["draw_patch"][f_idx]:
            patch = patches.Circle(
                (0, 0), radius=min_max_val, transform=ax.transData, fc="none", ec="k", lw=0.5
            )
            im.set_clip_path(patch)
            ax.add_patch(patch)
        ax.add_patch(plt.Circle((0, 0), radius=15, color="gray", fill=False, lw=0.5))

        # The old algorithm returns y_vals which corresponds to Y GSM from -min_max to +min_max
        y_vals_target = y_vals[f_idx]
        x_grid_old = np.linspace(-min_max_val, min_max_val, len(y_vals_target))
        
        def rolling_average(arr, window):
            if window < 2:
                return arr
            res = np.full_like(arr, np.nan, dtype=float)
            for i in range(len(arr)):
                start = max(0, i - window // 2)
                end = min(len(arr), i + window // 2 + 1)
                res[i] = np.nanmean(arr[start:end])
            return res
            
        y_vals_target_smooth = rolling_average(y_vals_target, SMOOTHING_POINTS_ORIGINAL)
        y_rad_smooth = rolling_average(y_rad, SMOOTHING_POINTS_BISECTION)
        z_rad_smooth = rolling_average(z_rad, SMOOTHING_POINTS_BISECTION)
        
        mms_y = sw_params["mms_sc_pos"][1]
        mms_z = sw_params["mms_sc_pos"][2]
        
        ivp_color = 'w' if figure_inputs["dark_mode"] else 'k'
        dist_texts = ["MMS Distances [$R_E$]:"]

        if figure_inputs.get("plot_orig_xline", True):
            idx_orig = np.nanargmin((x_grid_old - mms_y)**2 + (y_vals_target_smooth - mms_z)**2)
            dist_orig = np.sqrt((x_grid_old[idx_orig] - mms_y)**2 + (y_vals_target_smooth[idx_orig] - mms_z)**2)
            closest_orig_y, closest_orig_z = x_grid_old[idx_orig], y_vals_target_smooth[idx_orig]
            ax.plot(x_grid_old, y_vals_target_smooth, 'c-', linewidth=2, label='Original')
            ax.plot([mms_y, closest_orig_y], [mms_z, closest_orig_z], 'c--', linewidth=1.5)
            dist_texts.append(f"Original: {dist_orig:.2f}")

        if figure_inputs.get("plot_bisec_xline", True):
            idx_bisec = np.nanargmin((y_rad_smooth - mms_y)**2 + (z_rad_smooth - mms_z)**2)
            dist_bisec = np.sqrt((y_rad_smooth[idx_bisec] - mms_y)**2 + (z_rad_smooth[idx_bisec] - mms_z)**2)
            closest_bisec_y, closest_bisec_z = y_rad_smooth[idx_bisec], z_rad_smooth[idx_bisec]
            ax.plot(y_rad_smooth, z_rad_smooth, 'r-', linewidth=2, label='Bisection')
            ax.plot([mms_y, closest_bisec_y], [mms_z, closest_bisec_z], 'r--', linewidth=1.5)
            dist_texts.append(f"Bisection: {dist_bisec:.2f}")

        if figure_inputs.get("plot_ivp_xline", True):
            idx_ivp = np.nanargmin((y_ivp - mms_y)**2 + (z_ivp - mms_z)**2)
            dist_ivp = np.sqrt((y_ivp[idx_ivp] - mms_y)**2 + (z_ivp[idx_ivp] - mms_z)**2)
            closest_ivp_y, closest_ivp_z = y_ivp[idx_ivp], z_ivp[idx_ivp]
            ax.plot(y_ivp, z_ivp, color=ivp_color, linestyle='-', linewidth=2, label='IVP')
            ax.plot([mms_y, closest_ivp_y], [mms_z, closest_ivp_z], color=ivp_color, linestyle='--', linewidth=1.5)
            dist_texts.append(f"IVP: {dist_ivp:.2f}")

        ax.plot([y_start], [z_start], 'wo', markersize=8)
        ax.plot([mms_y], [mms_z], marker='$\\oplus$', markersize=15, color='white', markeredgecolor='white')

        # Plot a horizontal line at x=0 and a vertical line at y=0
        ax.axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)

        if figure_inputs["dark_mode"]:
            label_color = "w"
            clabel_color = "w"
            mtick_color = "w"
            text_color = "w"
        else:
            label_color = "k"
            clabel_color = "k"
            mtick_color = "k"
            text_color = "k"

        if f_idx == 0 or f_idx == 2:
            ax.set_ylabel(r"Z [GSM, $R_{\rm E}$]", fontsize=label_size, color=label_color)
        if f_idx == 2 or f_idx == 3:
            ax.set_xlabel(r"Y [GSM, $R_{\rm E}$]", fontsize=label_size, color=label_color)
        if f_idx == 1 or f_idx == 3:
            ax.set_ylabel(r"Z [GSM, $R_{\rm E}$]", fontsize=label_size, color=label_color)
            ax.yaxis.set_label_position("right")
            
        # Display distances in top-left corner
        if len(dist_texts) > 1:
            text_str = "\n".join(dist_texts)
            props = dict(boxstyle='round', facecolor='gray', alpha=0.7, edgecolor='white', linewidth=0.5)
            ax.text(0.03, 0.97, text_str, transform=ax.transAxes, fontsize=l_label_size,
                    verticalalignment='top', bbox=props, color=text_color)
                
        cax1 = divider1.append_axes("top", size="5%", pad=0.01)
        cbar1 = plt.colorbar(
            im, cax=cax1, orientation="horizontal", ticks=None, fraction=0.05, pad=0.01
        )
        cbar1.ax.tick_params(
            axis="x", direction="in", top=True, labeltop=True, bottom=False, labelbottom=False,
            pad=0.01, labelsize=ct_tick_size, labelcolor=label_color,
        )
        cbar1.ax.xaxis.set_label_position("top")
        cbar1.ax.set_xlabel(figure_inputs["c_label"][f_idx], fontsize=c_label_size, color=clabel_color)

        if f_idx == 0 or f_idx == 2:
            ax.tick_params(axis="both", direction="in", which="major", left=True, right=True, top=True, bottom=True,
                labelleft=True, labelright=False, labeltop=False, labelbottom=True,
                labelsize=t_label_size, length=tick_len, width=tick_width, labelcolor=label_color)
        else:
            ax.tick_params(axis="both", direction="in", which="major", left=True, right=True, top=True, bottom=True,
                labelleft=False, labelright=True, labeltop=False, labelbottom=True,
                labelsize=t_label_size, length=tick_len, width=tick_width, labelcolor=label_color)

        ax.minorticks_on()
        ax.tick_params(axis="both", which="minor", direction="in", length=mtick_len, left=True, right=True, top=True, bottom=True, color=mtick_color, width=mtick_width)
        ax.set_xticks([-16, -8, 0, 8, 16])
        ax.set_yticks([-16, -8, 0, 8, 16])
        plt.setp(ax.get_xticklabels(), rotation=0, ha="right", va="top", visible=True)
        plt.setp(ax.get_yticklabels(), rotation=0, va="center", visible=True)
        
        # Legend outside the plot area or inside, making sure it's readable
        # leg = ax.legend(bbox_to_anchor=(1.01, 1), loc='upper right' if f_idx%2==0 else 'upper left', borderaxespad=0.1, fontsize=12, labelspacing=0.5, handlelength=2, frameon=True, facecolor='gray', edgecolor='white', framealpha=0.3)
        # for text in leg.get_texts():
        #     text.set_color("white" if figure_inputs["dark_mode"] else "black")
            
        mms_sc_pos = figure_inputs["mms_sc_pos"]
        tsy_model = figure_inputs["tsy_model"]
        imf_clock_angle = figure_inputs["imf_clock_angle"]
        dipole_tilt_angle = figure_inputs["dipole_tilt_angle"]
        if figure_inputs["dark_mode"]:
            text_box_style = dict(boxstyle="round", facecolor="k", edgecolor="none", linewidth=0, alpha=0.3)
        else:
            text_box_style = dict(boxstyle="round", facecolor="w", edgecolor="none", linewidth=0, alpha=0.3)

        if f_idx == 0:
            ax.text(-0.18, 1.12, f"MMS Position - [{mms_sc_pos[0]:.2f}, {mms_sc_pos[1]:.2f}, {mms_sc_pos[2]:.2f}] $R_E$ \n [GSM]", horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes, rotation=0, color=text_color, fontsize=l_label_size, bbox=text_box_style)
        elif f_idx == 1:
            ax.text(1.15, 1.12, f"Model: {tsy_model.upper()}", horizontalalignment="right", verticalalignment="bottom", transform=ax.transAxes, rotation=0, color=text_color, fontsize=l_label_size, bbox=text_box_style)
        elif f_idx == 2:
            ax.text(-0.17, -0.1, f"Clock Angle: {imf_clock_angle:.2f}$^\\circ$", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, rotation=0, color=text_color, fontsize=l_label_size, bbox=text_box_style)
        elif f_idx == 3:
            ax.text(1.17, -0.1, f"Dipole tilt: {dipole_tilt_angle * 180 / np.pi:.2f}$^\\circ$", horizontalalignment="right", verticalalignment="top", transform=ax.transAxes, rotation=0, color=text_color, fontsize=l_label_size, bbox=text_box_style)

    import datetime
    t_range = figure_inputs["t_range"]
    if len(t_range) == 1:
        if isinstance(t_range[0], datetime.datetime):
            t_range_date = t_range[0]
        else:
            t_range_date = datetime.datetime.strptime(t_range[0], "%Y-%m-%d %H:%M:%S")
        dt = 5 # default dt
        t_range_date_min = t_range_date - datetime.timedelta(minutes=dt)
        t_range_date_max = t_range_date + datetime.timedelta(minutes=dt)
        t_range = [
            t_range_date_min.strftime("%Y-%m-%d %H:%M:%S"),
            t_range_date_max.strftime("%Y-%m-%d %H:%M:%S"),
        ]
        
    b_imf = figure_inputs["b_imf"]
    fig.suptitle(
        f"Time range: {t_range[0]} - {t_range[1]} \n $B_{{\\rm {{imf}}}}$ = [{b_imf[0]:.2f}, {b_imf[1]:.2f}, {b_imf[2]:.2f}] nT",
        fontsize=label_size,
        color=text_color,
        y=figure_inputs["title_y_pos"],
        alpha=0.65,
    )
        
    if figure_inputs["save_fig"]:
        from dateutil import parser
        import os
        from pathlib import Path
        temp1 = parser.parse(t_range[1]).strftime("%Y-%m-%d_%H-%M-%S")
        fig_time_range = f"{parser.parse(t_range[0]).strftime('%Y-%m-%d_%H-%M-%S')}_{temp1}"
        tsy_model = figure_inputs["tsy_model"]
        interpolation = figure_inputs["interpolation"]
        mms_probe_num = sw_params.get("mms_probe_num", "1")
        fig_version = figure_inputs["fig_version"]
        fig_folder = Path(
            f"figures/all_ridge_plots/{tsy_model}/{interpolation}"
            f"_interpolation_mms{mms_probe_num}/{fig_version}"
        )
        if not fig_folder.exists():
            fig_folder.mkdir(parents=True, exist_ok=True)
        
        bbb = f"{b_imf[0]:.0f}_{b_imf[1]:.0f}_{b_imf[2]:.0f}"
        fig_format = figure_inputs["fig_format"]
        fig_name = fig_folder / f"ridge_plot_{fig_time_range}_{bbb}_compare.{fig_format}"
        plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.05, format=fig_format, dpi=200)
        print(f"Saved plot to {os.path.abspath(fig_name)}")
        
if __name__ == '__main__':
    main()
