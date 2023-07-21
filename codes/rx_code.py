# This the python version of IDL code named "RX_model_batch.pro"
import datetime
import importlib
import os
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rx_model_funcs as rmf

importlib.reload(rmf)

# Set the fontstyle to Times New Roman
font = {"family": "serif", "weight": "normal", "size": 10}
plt.rc("font", **font)
plt.rc("text", usetex=True)

start = time.time()

today_date = datetime.datetime.today().strftime("%Y-%m-%d")

df_jet_reversal = pd.read_csv("../data/mms_jet_reversal_times_list_20230721_brst.csv",
                              index_col=False)
# If any column has NaN, drop that row
df_jet_reversal = df_jet_reversal.dropna()

# Drop rows which has same "jet_time" within 1 minute of each other
df_jet_reversal = df_jet_reversal.drop_duplicates(subset="jet_time", keep="first")
# Set the index to Date in UTC
df_jet_reversal.set_index("jet_time", inplace=True)
# Sort the dataframe by the index
df_jet_reversal.sort_index(inplace=True)
# Convert the index to datetime
df_jet_reversal.index = pd.to_datetime(df_jet_reversal.index)

time_diff = (df_jet_reversal.index[1:] - df_jet_reversal.index[:-1]).total_seconds()

# Drop rows where time_diff is less than 60 seconds
df_jet_reversal = df_jet_reversal.drop(df_jet_reversal.index[np.where(time_diff < 10)[0] + 1])

trange_list = df_jet_reversal.index.tolist()
mms_probe_num_list = [1, 2, 3, 4]

# Ask the user for the minmum and maximum index number
ind_min = int(input("Enter the minimum index number: "))
ind_max = int(input("Enter the maximum index number: "))


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


for mms_probe_num in mms_probe_num_list[2:3]:
    for ind_range, trange in enumerate(trange_list[ind_min:ind_max], start=ind_min):
        # Convert trange to string to format "%Y-%m-%d %H:%M:%S"
        trange = trange.strftime("%Y-%m-%d %H:%M:%S")
        trange = [trange.split("+")[0].split(".")[0]]

        print(f"\033[92m \n Started process for Figure number {ind_range} \033[0m \n")
        with suppress_stdout_stderr():
            try:
                mms_probe_num = str(mms_probe_num)
                min_max_val = 20
                dr = 0.25
                y_min = - min_max_val
                y_max = min_max_val
                z_min = - min_max_val
                z_max = min_max_val
                model_type = "t96"

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
                (bx, by, bz, shear, rx_en, va_cs, bisec_msp, bisec_msh, sw_params, x_shu, y_shu,
                 z_shu, b_msx, b_msy, b_msz) = rmf.rx_model(**model_inputs)

                # Normalize each quantity to the range [0, 1]
                shear_norm = (shear - np.nanmin(shear)) / (np.nanmax(shear) - np.nanmin(shear))
                rx_en_norm = (rx_en - np.nanmin(rx_en)) / (np.nanmax(rx_en) - np.nanmin(rx_en))
                va_cs_norm = (va_cs - np.nanmin(va_cs)) / (np.nanmax(va_cs) - np.nanmin(va_cs))
                bisec_msp_norm = (bisec_msp - np.nanmin(bisec_msp)) / (np.nanmax(bisec_msp) -
                                                                       np.nanmin(bisec_msp))
                bisec_msh_norm = (bisec_msh - np.nanmin(bisec_msh)) / (np.nanmax(bisec_msh) -
                                                                       np.nanmin(bisec_msh))

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
                    "sigma": [2, 2, 2, 2],
                    "mode": "nearest",
                    "alpha": 1,
                    "vmin": [0, 0, 0, 0],
                    "vmax": [1, 1, 1, 1],
                    "cmap_list": ["viridis", "cividis", "plasma", "magma"],
                    "draw_patch": [True, True, True, True],
                    "draw_ridge": [True, True, True, True],
                    "save_fig": True,
                    "fig_name": "crossing_all_ridge_plots",
                    "fig_format": "png",
                    "c_label": ["Shear", "Reconnection Energy", "Exhaust Velocity",
                                "Bisection Field"],
                    "wspace": 0.0,
                    "hspace": 0.17,
                    "fig_size": (8.775, 10),
                    "box_style": dict(boxstyle="round", color="k", alpha=0.8),
                    "title_y_pos": 1.09,
                    "interpolation": "None",
                    "tsy_model": model_type,
                    "dark_mode": False,
                    "rc_file_name": f"reconnection_line_data_mms{mms_probe_num}_20230721_msh.csv",
                    "rc_folder": "../data/rx_d/",
                    "save_rc_file": True,
                    "fig_version": "v_20230721",
                }

                y_vals, x_intr_vals_list, y_intr_vals_list = rmf.ridge_finder_multiple(
                                                            **figure_inputs)
                print(f"\033[92m \n Everything saved for Figure number {ind_range} \033[0m \n")
            except Exception as e:
                print(f"\033[91m \n Figure not plotted for time range {trange} \n because of"
                      f"following exception: {e} \n \033[0m")
                continue

print(f"Took {round(time.time() - start, 3)} seconds")
