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
)
import rxn_location.master_jet_list as mjl

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch process jet reversal checks and reconnection models from a list of times."
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
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging output level. Default: INFO"
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
        ["Log Level", args.log_level],
    ]
    print("\n" + "="*60)
    print("RXN Location: Batch Statistics Processing")
    print("="*60)
    print(tabulate(param_table, headers=["Parameter", "Value"], tablefmt="fancy_grid"))
    print("="*60 + "\n")
    
    # Configure root logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
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
    
    for idx, c_time in enumerate(file_times):
        time_str = c_time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{idx+1}/{len(file_times)}] Processing: {time_str}")
        
        # 1. Jet Reversal Check
        try:
            # Note: the jet_reversal_check function expects a datetime object or a specific string
            # we pass datetime object directly
            s1, det = jet_reversal_check(
                c_time,
                mms_probe_num=args.probe,
                data_rate=args.data_rate,
                save_fig=True,
                fig_name=f"jet_reversal_{time_str.replace(' ', '_').replace(':', '')}",
                fig_format=args.format,
            )
        except Exception as e:
            print(f"  -> Error running Jet Check: {e}")
            continue
            
        if not s1 or det is None:
            print(f"  -> No jet found at {time_str}.")
            continue
            
        print(f"  -> Jet detected!")
        
        # 2. Reconnection Models
        try:
            model_inputs = {
                "trange": [time_str],
                "probe": None,
                "omni_level": "hro",
                "mms_probe_num": args.probe,
                "model_type": args.tsy_model,
                "m_p": 0.5,
                "dr": 0.25,
            }
            images, c_labels = [], []
            model_mapping = {
                "shear": {"var_idx": 3, "label": "Shear"},
                "bisection": {"var_idx": 6, "label": "Bisection Field"},
                "reconnection energy": {"var_idx": 4, "label": "Reconnection Energy"},
                "exhaust velocity": {"var_idx": 5, "label": "Exhaust Velocity"},
            }
            
            for m in models_to_run:
                model_inputs["var_idx"] = model_mapping[m]["var_idx"]
                images.append(model_inputs.copy())
                c_labels.append(model_mapping[m]["label"])
                
            figure_inputs = {
                "images": images,
                "imf_clock_angle": [0],
                "v_imf": [0, 0, -400],
                "b_imf": [0, 0, -5],
                "p_dyn": [2],
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
                "tsy_model": args.tsy_model,
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
            "dt": 60,
            "jet_len": 3,
            "data_rate": args.data_rate,
            "level": "l2",
            "coord_type": "lmn",
            "time_clip": True,
            "t_delta": 10,
            "max_attempts": 5,
            "tsy_model": args.tsy_model,
            "recon_models": models_to_run,
            "omni_level": "hro",
            "m_p": 0.5,
            "dr": 0.25,
            "limits": 15,
        }
        
        # Pull latest OMNI SW params for contextual Master List logging
        try:
            sw_params = get_sw_params(
                trange=[time_str], omni_level="hro", mms_probe_num=args.probe
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

    print(f"\nBatch processing complete! Output saved to {args.outdir}/")

if __name__ == "__main__":
    main()
