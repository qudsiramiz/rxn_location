import sys
import datetime
from rxn_location.jet_reversal_check_function import jet_reversal_check
from rxn_location.rx_model_funcs import ridge_finder_multiple, rx_model
import numpy as np
import pytz

c_time = datetime.datetime(2015, 9, 2, 16, 45, 0, tzinfo=pytz.utc)
time_str = "2015-09-02 16:45:00"

try:
    fig, s1, det = jet_reversal_check(
        c_time,
        probe=3,
        data_rate="fast",
        dt=300,
        jet_len=3,
    )

    model_inputs = {
        "trange": [time_str],
        "probe": None,
        "omni_level": "hro",
        "mms_probe_num": 3,
        "model_type": "t96",
        "m_p": 0.5,
        "dr": 0.25,
    }
    
    res = rx_model(**model_inputs)
    sw_params = res[8]
    images, c_labels = [], []
    model_mapping = {
        "shear": {"var_idx": 3, "label": "Shear"},
    }
    
    for m in ["shear"]:
        var_idx = model_mapping[m]["var_idx"]
        raw_data = res[var_idx]
        norm_data = (raw_data - np.nanmin(raw_data)) / (np.nanmax(raw_data) - np.nanmin(raw_data))
        images.append(norm_data)
        c_labels.append(model_mapping[m]["label"])
        
    figure_inputs = {
        "image": images,
        "convolution_order": [1] * len(images),
        "t_range": [time_str],
        "b_imf": np.round(sw_params["b_imf"], 2),
        "b_msh": np.round(sw_params["mms_b_gsm"], 2),
        "xrange": [-15, 15],
        "yrange": [-15, 15],
        "mms_probe_num": str(3),
        "mms_sc_pos": np.round(sw_params["mms_sc_pos"], 2),
        "dr": 0.25,
        "dipole_tilt_angle": sw_params["ps"],
        "p_dyn": np.round(sw_params["p_dyn"], 2),
        "imf_clock_angle": sw_params["imf_clock_angle"],
        "np_imf": np.round(sw_params["np"], 2),
        "v_imf": np.round(sw_params["v_imf"], 2),
        "sym_h": np.round(sw_params["sym_h"], 2),
        "sigma": [2] * len(images),
        "mode": "nearest",
        "alpha": 1,
        "vmin": [0] * len(images),
        "vmax": [1] * len(images),
        "cmap_list": ["viridis", "cividis", "plasma", "magma"][:len(images)],
        "draw_patch": [True] * len(images),
        "draw_ridge": [True] * len(images),
        "save_fig": True,
        "fig_name": f"recon_models_{time_str.replace(' ', '_').replace(':', '')}",
        "fig_format": "png",
        "c_label": c_labels,
        "wspace": 0.15,
        "hspace": 0.17,
        "fig_size": (10, 8),
        "box_style": dict(boxstyle="round", color="k", alpha=0.8),
        "title_y_pos": 1.09,
        "interpolation": "None",
        "tsy_model": "t96",
        "dark_mode": False,
        "save_rc_file": False,
        "rc_file_name": "test.csv",
        "rc_folder": "./",
        "df_jet_reversal": det,
    }
    
    print("Testing ridge_finder_multiple")
    _, _, _, _ = ridge_finder_multiple(**figure_inputs)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
