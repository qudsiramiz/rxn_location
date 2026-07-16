import argparse
import datetime
import pytz
import os
import sys
import logging
from dateutil import parser
from tabulate import tabulate

# Ensure the parent directory is in sys.path so we can import rxn_location
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rxn_location.jet_reversal_check_function import jet_reversal_check
from rxn_location.rx_model_funcs import (
    ridge_finder_multiple,
    ridge_finder_multiple_interactive,
    get_sw_params,
    rx_model,
)
import numpy as np
import rxn_location.master_jet_list as mjl

def parse_args():
    class CommentedArgumentParser(argparse.ArgumentParser):
        def convert_arg_line_to_args(self, arg_line):
            arg_line = arg_line.split('#')[0].strip()
            if arg_line:
                yield arg_line

    parser = CommentedArgumentParser(
        description="Batch process jet reversal checks and reconnection models from a list of times.",
        fromfile_prefix_chars="@"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to input .txt or .csv file with a list of times."
    )
    parser.add_argument(
        "--probe", type=str, default="3", help="MMS probe number (1-4). Default: 3"
    )
    parser.add_argument(
        "--tsy_model", type=str, default="T96", help="Tsyganenko model to use. Default: T96"
    )
    parser.add_argument(
        "--data_rate", type=str, default="fast", help="MMS data rate (fast/brst). Default: fast"
    )
    parser.add_argument(
        "--format", choices=["html", "pdf", "png"], default="html", 
        help="Format for saved plots (html for interactive Plotly, pdf/png for static Matplotlib). Default: html"
    )
    parser.add_argument(
        "--outdir", type=str, default="./figures", help="Output directory for plots. Default: ./figures"
    )
    parser.add_argument(
        "--csv_name", type=str, default="batch_reconnection_stats.csv", help="Name of output stats CSV."
    )
    parser.add_argument(
        "--dt", type=int, default=300, help="Time window around crossing time to search for jets (s). Default: 300"
    )
    parser.add_argument(
        "--jet_len", type=int, default=3, help="Minimum number of data points for a valid jet. Default: 3"
    )
    parser.add_argument(
        "--m_p", type=float, default=0.5, help="Magnetopause standoff distance scaling. Default: 0.5"
    )
    parser.add_argument(
        "--dr", type=float, default=0.25, help="Spatial resolution for 3D model grid (Re). Default: 0.25"
    )
    parser.add_argument(
        "--limits", type=int, default=20, help="X, Y, Z boundaries for 3D grid (Re). Default: 20"
    )
    parser.add_argument(
        "--log-level", type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging output level. Default: WARNING"
    )
    parser.add_argument(
        "--t_delta", type=int, default=2,
        help="Minutes to shift the crossing time on each retry when searching for jets. Default: 2"
    )
    parser.add_argument(
        "--max_retries", type=int, default=5,
        help="Maximum number of time-shifted retries when initial jet check fails. Default: 5"
    )
    parser.add_argument(
        "--plot-seaborn",
        action="store_true",
        help="Generate final statistical Seaborn joint-plots from the Master List.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Print default/passed parameters using Tabulate
    param_table = [
        ["Input File", args.input],
        ["MMS Probe", args.probe],
        ["Tsyganenko Model", args.tsy_model],
        ["Data Rate", args.data_rate],
        ["Plot Format", args.format],
        ["Output Directory", args.outdir],
        ["Stats CSV Name", args.csv_name],
        ["Jet Window (dt)", args.dt],
        ["Jet Length Threshold", args.jet_len],
        ["MP Standoff (m_p)", args.m_p],
        ["Grid Resolution (dr)", args.dr],
        ["Grid Limits", args.limits],
        ["Log Level", args.log_level],
        ["Retry Δt (min)", args.t_delta],
        ["Max Retries", args.max_retries],
    ]
    print("\n" + "="*60)
    print("RXN Location: Batch Statistics Processing")
    print("="*60)
    print(tabulate(param_table, headers=["Parameter", "Value"], tablefmt="fancy_grid"))
    print("="*60 + "\n")
    
    # Configure root logging
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    
    # Suppress verbose pyspedas/pytplot logs unless DEBUG is requested
    if args.log_level != "DEBUG":
        logging.getLogger("pyspedas").setLevel(logging.WARNING)
        logging.getLogger("pytplot").setLevel(logging.WARNING)
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
        
    os.makedirs(args.outdir, exist_ok=True)
    
    # Read times
    file_times = []
    with open(args.input, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(","):
                continue
            first_col = line.split(",")[0]
            try:
                t = parser.parse(first_col)
                if t.tzinfo is None:
                    t = t.replace(tzinfo=pytz.utc)
                file_times.append(t)
            except Exception:
                pass
                
    if not file_times:
        print("Error: No valid timestamps found in the input file.")
        sys.exit(1)
        
    print(f"Found {len(file_times)} valid timestamps to process.\n")
    
    master_jets = mjl.load_master_list()
    
    # Define models to run
    models_to_run = ["shear", "bisection", "reconnection energy", "exhaust velocity"]
    
    for idx, c_time_original in enumerate(file_times):
        time_str = c_time_original.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{idx+1}/{len(file_times)}] Processing: {time_str}")
        
        # 1. Jet Reversal Check (with retry logic)
        # Try the original time first, then shift by ±t_delta minutes
        fig, s1, det = None, False, None
        c_time = c_time_original  # the time that actually found the jet
        
        # Build list of times to try: original, then ±1*delta, ±2*delta, ...
        times_to_try = [c_time_original]
        for i in range(1, args.max_retries + 1):
            times_to_try.append(c_time_original + datetime.timedelta(minutes=i * args.t_delta))
            times_to_try.append(c_time_original - datetime.timedelta(minutes=i * args.t_delta))
        
        for attempt_idx, attempt_time in enumerate(times_to_try):
            try:
                fig_try, s1_try, det_try = jet_reversal_check(
                    attempt_time,
                    probe=args.probe,
                    data_rate=args.data_rate,
                    dt=args.dt,
                    jet_len=args.jet_len,
                )
            except Exception as e:
                if attempt_idx == 0:
                    print(f"  -> Error running Jet Check: {e}")
                continue
            
            if s1_try and det_try is not None:
                fig, s1, det = fig_try, s1_try, det_try
                c_time = attempt_time
                if attempt_idx > 0:
                    print(f"  -> Jet found at shifted time {attempt_time.strftime('%Y-%m-%d %H:%M:%S')} "
                          f"(attempt {attempt_idx + 1}/{len(times_to_try)})")
                break
            elif attempt_idx == 0:
                print(f"  -> No jet at original time, searching nearby times (±{args.t_delta}min steps)...")
        
        if not s1 or det is None:
            print(f"  -> No jet found at {time_str} or within ±{args.max_retries * args.t_delta}min window.")
            continue
            
        print(f"  -> Jet detected!")
        
        # Save the jet plot
        if fig is not None:
            time_str_safe = time_str.replace(" ", "_").replace(":", "")
            jet_plot_filename = os.path.join(args.outdir, f"jet_plot_{time_str_safe}.{args.format}")
            try:
                if args.format == "html":
                    fig.write_html(jet_plot_filename)
                else:
                    fig.write_image(jet_plot_filename)
                print(f"  -> Saved Jet Plot to {jet_plot_filename}")
            except Exception as e:
                print(f"  -> Failed to save Jet Plot: {e}")
        
        # 2. Reconnection Models (use actual jet time, which may differ from original)
        jet_time_str = c_time.strftime('%Y-%m-%d %H:%M:%S')
        try:
            model_inputs = {
                "trange": [jet_time_str],
                "probe": None,
                "omni_level": "hro",
                "mms_probe_num": args.probe,
                "model_type": args.tsy_model.lower(),
                "m_p": args.m_p,
                "dr": args.dr,
                "min_max_val": args.limits,
            }
            
            res = rx_model(**model_inputs)
            if not res:
                print(f"  -> Model generation failed for {time_str}")
                continue
                
            sw_params = res[8]
            
            images, c_labels = [], []
            model_mapping = {
                "shear": {"var_idx": 3, "label": "Shear"},
                "bisection": {"var_idx": 6, "label": "Bisection Field"},
                "reconnection energy": {"var_idx": 4, "label": "Reconnection Energy"},
                "exhaust velocity": {"var_idx": 5, "label": "Exhaust Velocity"},
            }
            
            for m in models_to_run:
                var_idx = model_mapping[m]["var_idx"]
                raw_data = res[var_idx]
                norm_data = (raw_data - np.nanmin(raw_data)) / (np.nanmax(raw_data) - np.nanmin(raw_data))
                images.append(norm_data)
                c_labels.append(model_mapping[m]["label"])
                
            figure_inputs = {
                "image": images,
                "convolution_order": [1] * len(images),
                "t_range": [jet_time_str],
                "b_imf": np.round(sw_params["b_imf"], 2),
                "b_msh": np.round(sw_params["mms_b_gsm"], 2),
                "xrange": [-args.limits, args.limits],
                "yrange": [-args.limits, args.limits],
                "mms_probe_num": str(args.probe),
                "mms_sc_pos": np.round(sw_params["mms_sc_pos"], 2),
                "dr": args.dr,
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
                "fig_format": args.format,
                "c_label": c_labels,
                "wspace": 0.15,
                "hspace": 0.17,
                "fig_size": (8.775, 10) if args.format == "html" else (10, 8),
                "box_style": dict(boxstyle="round", color="k", alpha=0.8),
                "title_y_pos": 1.09,
                "interpolation": "None",
                "tsy_model": args.tsy_model.lower(),
                "dark_mode": False,
                "save_rc_file": True,
                "rc_file_name": args.csv_name,
                "rc_folder": args.outdir,
                "df_jet_reversal": det,
            }
            
            print("  -> Generating Reconnection Models...")
            if args.format == "html":
                _ = ridge_finder_multiple_interactive(**figure_inputs)
            else:
                _ = ridge_finder_multiple(**figure_inputs)
                
            print("  -> Models generated and saved!")
            
        except Exception as e:
            print(f"  -> Error running Models: {e}")
            continue

        # 3. Save to Master List
        run_params = {
            "mms_probe": int(args.probe),
            "dt": args.dt,
            "jet_len": args.jet_len,
            "data_rate": args.data_rate,
            "level": "l2",
            "coord_type": "lmn",
            "time_clip": True,
            "t_delta": 10,
            "max_attempts": 5,
            "tsy_model": args.tsy_model,
            "recon_models": models_to_run,
            "omni_level": "hro",
            "m_p": args.m_p,
            "dr": args.dr,
            "limits": args.limits,
        }
        
        # Pull latest OMNI SW params for contextual Master List logging
        try:
            sw_params = get_sw_params(
                trange=[jet_time_str], omni_level="hro", mms_probe_num=args.probe
            )
            if sw_params is not None:
                det["sw_b_imf_gsm_x"] = sw_params["b_imf"][0]
                det["sw_b_imf_gsm_y"] = sw_params["b_imf"][1]
                det["sw_b_imf_gsm_z"] = sw_params["b_imf"][2]
                det["sw_v_imf_gse_x"] = sw_params["v_imf"][0]
                det["sw_v_imf_gse_y"] = sw_params["v_imf"][1]
                det["sw_v_imf_gse_z"] = sw_params["v_imf"][2]
                det["sw_np"] = sw_params["np"]
                det["sw_tp"] = sw_params["t_p"]
                det["sw_sym_h"] = sw_params["sym_h"]
                det["sw_clock_angle"] = sw_params["imf_clock_angle"]
                det["sw_p_dyn"] = sw_params["p_dyn"]
                
                # Compute cone angle
                import math
                bx, by, bz = sw_params["b_imf"][0], sw_params["b_imf"][1], sw_params["b_imf"][2]
                b_mag = math.sqrt(bx**2 + by**2 + bz**2)
                if b_mag > 0:
                    det["sw_cone_angle"] = math.acos(bx / b_mag) * 180 / math.pi
        except Exception:
            pass 
            
        was_added, existing = mjl.add_jet(
            master_jets,
            det,
            c_time,
            run_params,
            window_minutes=2
        )
        
        if was_added:
            mjl.save_master_list(master_jets)
            print("  -> Logged to Master Jet List.")
        else:
            print(f"  -> Jet already exists in Master List (near {existing.get('jet_time')}).")

    if args.plot_seaborn:
        print("\nGenerating Statistical Seaborn Plots...")
        df_stats = mjl.master_list_to_stats_csv(master_jets)
        if df_stats is not None and not df_stats.empty:
            from rxn_location.app_seaborn_plots import generate_seaborn_jointplots
            try:
                fig = generate_seaborn_jointplots(
                    df_stats, 
                    x_key="b_imf_z", 
                    y_key="b_imf_y", 
                    x_label=r"IMF $B_{\rm z}$ [nT]", 
                    y_label=r"IMF $B_{\rm y}$ [nT]", 
                    marker_size_var="r_rc"
                )
                fig_path = os.path.join(args.outdir, "seaborn_stats_By_vs_Bz.png")
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                print(f"  -> Saved Seaborn plot to {fig_path}")
            except Exception as e:
                print(f"  -> Error generating Seaborn plot: {e}")
        else:
            print("  -> Not enough data to generate Seaborn plots.")

    print(f"\nBatch processing complete! Output saved to {args.outdir}/")

if __name__ == "__main__":
    main()
