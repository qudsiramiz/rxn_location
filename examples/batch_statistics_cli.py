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
from rxn_location.logger import set_verbosity, vprint

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
        "--verbosity", type=int, default=2, choices=[0, 1, 2, 3],
        help="Verbosity level (0-3). 0: silent, 1: important only, 2: standard (no PySPEDAS), 3: all. Default: 2"
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
    parser.add_argument(
        "--plot_engine", type=str, default="plotly", choices=["plotly", "matplotlib"],
        help="Engine used to plot the jet reversal check (plotly or matplotlib). Default: plotly"
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
        ["Verbosity", args.verbosity],
        ["Retry Δt (min)", args.t_delta],
        ["Max Retries", args.max_retries],
    ]
    vprint(1, "\n" + "="*60, color="cyan")
    vprint(1, "RXN Location: Batch Statistics Processing", color="bold")
    vprint(1, "="*60, color="cyan")
    vprint(1, tabulate(param_table, headers=["Parameter", "Value"], tablefmt="fancy_grid"), color="cyan")
    vprint(1, "="*60 + "\n", color="cyan")
    
    # Configure global verbosity
    set_verbosity(args.verbosity)
    
    if not os.path.exists(args.input):
        vprint(1, f"Error: Input file '{args.input}' not found.", color="red")
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
        vprint(1, "Error: No valid timestamps found in the input file.", color="red")
        sys.exit(1)
        
    vprint(1, f"Found {len(file_times)} valid timestamps to process.\n", color="green")
    
    master_jets = mjl.load_master_list()
    
    # Define models to run
    models_to_run = ["shear", "reconnection energy", "exhaust velocity", "bisection"]
    
    for idx, c_time_original in enumerate(file_times):
        time_str = c_time_original.strftime('%Y-%m-%d %H:%M:%S')
        vprint(1, f"[{idx+1}/{len(file_times)}] Processing: {time_str}", color="blue")
        
        # Check if time is already in master list
        already_processed = False
        for entry in master_jets:
            try:
                ct = entry.get("crossing_time")
                if ct:
                    ct_dt = mjl._parse_time(ct)
                    if abs(ct_dt - c_time_original) <= datetime.timedelta(seconds=2):
                        already_processed = True
                        break
            except Exception:
                pass
                
        if not already_processed:
            existing = mjl.find_nearby_jet(master_jets, c_time_original, window_minutes=2)
            if existing is not None:
                already_processed = True

        if already_processed:
            vprint(1, f"  -> Skipping: Time {time_str} is already present in the Master List.", color="yellow")
            continue
        
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
            time_str_safe = attempt_time.strftime("%Y-%m-%d_%H%M%S")
            jet_plot_filename = os.path.join(args.outdir, f"jet_plot_{time_str_safe}.{args.format}")
            
            try:
                res = jet_reversal_check(
                    attempt_time,
                    probe=args.probe,
                    data_rate=args.data_rate,
                    dt=args.dt,
                    jet_len=args.jet_len,
                    return_plotly_fig=(args.plot_engine == "plotly"),
                    figname=jet_plot_filename if args.plot_engine == "matplotlib" else None,
                )
            except Exception as e:
                if attempt_idx == 0:
                    vprint(1, f"  -> Error running Jet Check: {e}", color="red")
                continue
            
            if res is None:
                continue
                
            fig_try, s1_try, det_try = res
            
            if s1_try and det_try is not None:
                fig, s1, det = fig_try, s1_try, det_try
                c_time = attempt_time
                if attempt_idx > 0:
                    vprint(1, f"  -> Jet found at shifted time {attempt_time.strftime('%Y-%m-%d %H:%M:%S')} "
                              f"(attempt {attempt_idx + 1}/{len(times_to_try)})", color="yellow")
                break
            elif attempt_idx == 0:
                vprint(2, f"  -> No jet at original time, searching nearby times (±{args.t_delta}min steps)...", color="magenta")
        
        if not s1 or det is None:
            vprint(1, f"  -> No jet found at {time_str} or within ±{args.max_retries * args.t_delta}min window.", color="red")
            continue
            
        vprint(1, f"  -> Jet detected!", color="green")
        
        # Save the jet plot
        if args.plot_engine == "plotly" and fig is not None:
            try:
                if args.format == "html":
                    fig.write_html(jet_plot_filename)
                else:
                    fig.write_image(jet_plot_filename)
                
                abs_jet_plot_filename = os.path.abspath(jet_plot_filename)
                vprint(2, f"  -> Saved Jet Plot to {abs_jet_plot_filename}", color="magenta")
            except Exception as e:
                vprint(1, f"  -> Failed to save Jet Plot: {e}", color="red")
        elif args.plot_engine == "matplotlib":
            abs_jet_plot_filename = os.path.abspath(jet_plot_filename)
            vprint(2, f"  -> Saved Jet Plot to {abs_jet_plot_filename}", color="magenta")
        
        # 2. Reconnection Models (use exact jet time from detection)
        exact_jet_time = det.get("jet_time")
        if exact_jet_time is None:
            exact_jet_time = c_time
        jet_time_str = exact_jet_time.strftime('%Y-%m-%d %H:%M:%S')
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
                vprint(1, f"  -> Model generation failed for {time_str}", color="red")
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
                "wspace": 0.0,
                "hspace": 0.20,
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
            
            vprint(2, "  -> Generating Reconnection Models...", color="magenta")
            if args.format == "html":
                _, dist_rc_dict = ridge_finder_multiple_interactive(**figure_inputs)
            else:
                _, _, _, dist_rc_dict = ridge_finder_multiple(**figure_inputs)
                
            if dist_rc_dict:
                det.update(dist_rc_dict)
                
            vprint(2, "  -> Models generated and saved!", color="magenta")
            
        except Exception as e:
            vprint(1, f"  -> Error running Models: {e}", color="red")
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
            trange_date_min = exact_jet_time - datetime.timedelta(minutes=30)
            trange_date_max = exact_jet_time + datetime.timedelta(minutes=30)
            trange_min_str = trange_date_min.strftime("%Y-%m-%d %H:%M:%S") + "Z"
            trange_max_str = trange_date_max.strftime("%Y-%m-%d %H:%M:%S") + "Z"
            
            sw_params = get_sw_params(
                trange=[trange_min_str, trange_max_str], omni_level="hro", mms_probe_num=str(args.probe)
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
            vprint(1, "  -> Logged to Master Jet List.", color="green")
        else:
            vprint(2, f"  -> Jet already exists in Master List (near {existing.get('jet_time')}).", color="yellow")

    if args.plot_seaborn:
        vprint(1, "\n" + "="*60, color="cyan")
        vprint(1, "Generating Seaborn joint-plots for master list parameters...", color="bold")
        vprint(1, "="*60, color="cyan")
        df_stats = mjl.master_list_to_stats_csv(master_jets)
        if df_stats is not None and not df_stats.empty:
            from rxn_location.app_seaborn_plots import generate_seaborn_jointplots
            try:
                _ = generate_seaborn_jointplots(df_full=master_jets, dark_mode=False)
                vprint(1, "  -> Done generating Seaborn plots.", color="green")
            except Exception as e:
                vprint(1, f"  -> Error generating Seaborn plot: {e}", color="red")
        else:
            vprint(1, "  -> Not enough data to generate Seaborn plots.", color="yellow")

    vprint(1, f"\nBatch processing complete! Output saved to {args.outdir}/", color="green")

if __name__ == "__main__":
    main()
