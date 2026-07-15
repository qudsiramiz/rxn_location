"""
Streamlit GUI Application for Reconnection Location Analysis

This module contains the interactive web interface for the `rxn_location` package.
It integrates Jet Reversal detection, 3D Reconnection model visualization,
and batch processing Statistics Mode for analyzing MMS spacecraft data.
"""

import streamlit as st
import streamlit.components.v1 as components
import datetime
import plotly.io as pio
import pandas as pd
import io
import zipfile
import pytz
import pickle
import os
from pathlib import Path

# Try importing the required project modules
try:
    from rxn_location.jet_reversal_check_function import jet_reversal_check
    from rxn_location.rx_model_funcs import rx_model, ridge_finder_multiple_interactive
    from rxn_location.app_stats_plots import generate_statistics_plots
except ImportError as e:
    st.error(
        f"Error importing rxn_location modules. Ensure the package is installed properly.\n{e}"
    )

# Page config
st.set_page_config(page_title="RXN Location GUI", layout="wide", initial_sidebar_state="expanded")

AUTO_SAVE_PATH = Path(os.path.expanduser("~")) / ".rxn_location_auto_save.pkl"


def save_auto_session():
    state_to_save = {
        "history": st.session_state.get("history", {"jet_checks": [], "recon_models": []}),
        "crossing_time_str": st.session_state.get("crossing_time_str", "2015-09-02 16:45:00"),
        "dark_mode": st.session_state.get("dark_mode", True),
    }
    try:
        with open(AUTO_SAVE_PATH, "wb") as f:
            pickle.dump(state_to_save, f)
    except Exception as e:
        pass  # silently fail on auto-save errors


def load_auto_session():
    if AUTO_SAVE_PATH.exists():
        try:
            with open(AUTO_SAVE_PATH, "rb") as f:
                saved_state = pickle.load(f)
            return saved_state
        except:
            return None
    return None


def reset_session():
    if AUTO_SAVE_PATH.exists():
        AUTO_SAVE_PATH.unlink()

    csv_to_remove = st.session_state.get("latest_stats_csv", "reconnection_stats.csv")
    if os.path.exists(csv_to_remove):
        os.remove(csv_to_remove)

    keys_to_clear = [
        "history",
        "crossing_time_str",
        "dark_mode",
        "initialized",
        "jet_view_idx",
        "rxn_view_idx",
        "latest_stats_csv",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]


def inject_beforeunload():
    st.iframe(
        """
        <script>
            window.addEventListener('beforeunload', function (e) {
                e.preventDefault();
                e.returnValue = '';
            });
        </script>
        """,
        height=1,
        width=1,
    )


def main():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("RXN Location Analyzer")
        st.markdown("Analyze Jet Reversals and Reconnection Models using MMS data.")
    with col2:
        top_counter_placeholder = st.empty()

    inject_beforeunload()

    # Initialization
    if "initialized" not in st.session_state:
        saved_state = load_auto_session()
        if saved_state:
            st.session_state["history"] = saved_state.get(
                "history", {"jet_checks": [], "recon_models": []}
            )
            st.session_state["crossing_time_str"] = saved_state.get(
                "crossing_time_str", "2015-09-02 16:45:00"
            )
            st.session_state["dark_mode"] = saved_state.get("dark_mode", True)
        else:
            st.session_state["history"] = {"jet_checks": [], "recon_models": []}
            st.session_state["crossing_time_str"] = "2015-09-02 16:45:00"
            st.session_state["dark_mode"] = True

        st.session_state["initialized"] = True
        st.session_state["jet_view_idx"] = -1
        st.session_state["rxn_view_idx"] = -1

    # Sidebar Session Manager
    with st.sidebar.expander("Session Manager", expanded=False):
        uploaded_file = st.file_uploader(
            "Restore Session (.rxn_session)", type=["rxn_session", "pkl"]
        )
        if uploaded_file is not None:
            if st.button("Load Session", use_container_width=True):
                try:
                    loaded_state = pickle.load(uploaded_file)
                    st.session_state["history"] = loaded_state.get(
                        "history", {"jet_checks": [], "recon_models": []}
                    )
                    st.session_state["crossing_time_str"] = loaded_state.get(
                        "crossing_time_str", "2015-09-02 16:45:00"
                    )
                    st.session_state["dark_mode"] = loaded_state.get("dark_mode", True)
                    st.session_state["jet_view_idx"] = -1
                    st.session_state["rxn_view_idx"] = -1
                    save_auto_session()
                    st.success("Session loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load session: {e}")

        # Download current session
        session_data = {
            "history": st.session_state["history"],
            "crossing_time_str": st.session_state["crossing_time_str"],
            "dark_mode": st.session_state["dark_mode"],
        }
        try:
            session_bytes = pickle.dumps(session_data)
            st.download_button(
                label="Download Current Session",
                data=session_bytes,
                file_name=f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.rxn_session",
                mime="application/octet-stream",
                use_container_width=True,
            )
        except Exception as e:
            st.error("Cannot serialize session.")

    # Sidebar inputs
    st.sidebar.header("Global Settings")

    if st.session_state.dark_mode:
        button_icon = "☀️ Light Mode"
    else:
        button_icon = "🌙 Dark Mode"

    if st.sidebar.button(button_icon):
        st.session_state.dark_mode = not st.session_state.dark_mode
        template = "plotly_dark" if st.session_state.dark_mode else "plotly_white"
        # Update all historical figures to match theme
        for run in st.session_state["history"]["jet_checks"]:
            if run.get("fig"):
                run["fig"].update_layout(template=template)
        for run in st.session_state["history"]["recon_models"]:
            if run.get("fig"):
                run["fig"].update_layout(template=template)
        save_auto_session()
        st.rerun()

    is_dark_mode = st.session_state.dark_mode

    show_hints = st.sidebar.checkbox(
        "Show Parameter Hints",
        value=False,
        help="Toggle to show tooltip hints on optional parameters.",
    )

    st.sidebar.header("Observation Parameters")

    date_str = st.sidebar.text_input(
        "Crossing Time (UTC)", value=st.session_state.crossing_time_str
    )
    try:
        crossing_time = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        crossing_time = crossing_time.replace(tzinfo=pytz.utc)
        st.session_state.crossing_time_str = date_str
    except Exception as e:
        st.sidebar.error("Invalid time format. Use YYYY-MM-DD HH:MM:SS")
        crossing_time = None

    mms_probe = st.sidebar.selectbox(
        "MMS Probe",
        [1, 2, 3, 4],
        index=2,
        help=(
            "Select which MMS spacecraft probe to use for the observation data."
            if show_hints
            else None
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Statistics Mode Parameters")
    use_stats_mode = st.sidebar.checkbox("Enable Statistics Mode")
    stop_condition = st.sidebar.radio(
        "Stop Condition", ["End Time", "Target Number of Jets"], disabled=not use_stats_mode
    )

    if stop_condition == "End Time":
        stats_end_time_str = st.sidebar.text_input(
            "End Time (UTC)",
            value=st.session_state.crossing_time_str,
            help="Only used if Statistics Mode is enabled." if show_hints else None,
            disabled=not use_stats_mode,
        )
        try:
            stats_end_time = datetime.datetime.strptime(
                stats_end_time_str, "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=pytz.utc)
        except:
            stats_end_time = crossing_time
        target_jets = None
        end_str_safe = stats_end_time_str.replace(" ", "_").replace(":", "")
    else:
        target_jets = st.sidebar.number_input(
            "Number of Jets to Find", min_value=1, value=10, disabled=not use_stats_mode
        )
        stats_end_time = None
        end_str_safe = f"{target_jets}jets"

    stats_delta_mins = st.sidebar.number_input(
        "Time Delta (minutes)",
        min_value=1,
        value=15,
        help="Step size for the batch loop." if show_hints else None,
        disabled=not use_stats_mode,
    )
    start_str_safe = date_str.replace(" ", "_").replace(":", "")
    default_csv_name = f"reconnection_stats_{start_str_safe}_{end_str_safe}_{stats_delta_mins}m.csv"
    stats_csv_name = st.sidebar.text_input(
        "Output CSV Name",
        value=default_csv_name,
        help="Name of the file to save batch data to." if show_hints else None,
        disabled=not use_stats_mode,
    )

    sidebar_actions = st.sidebar.container()  # placeholder for buttons

    # Create Tabs for the main display area
    if use_stats_mode:
        tab_controls, tab_jet, tab_rxn, tab_stats = st.tabs(
            ["Controls", "Jet Reversal Plot", "Reconnection Models", "Statistics Results"]
        )
    else:
        tab_controls, tab_jet, tab_rxn = st.tabs(
            ["Controls", "Jet Reversal Plot", "Reconnection Models"]
        )
        tab_stats = None

    with tab_jet:
        live_jet_plot_placeholder = st.empty()

    with tab_controls:
        # Indicator for Jet Reversal Status on Controls Tab
        if len(st.session_state["history"]["jet_checks"]) > 0:
            latest_jet = st.session_state["history"]["jet_checks"][-1]
            if latest_jet["jet_detection"]:
                st.success(
                    f"Latest Jet Check ({latest_jet['crossing_time']}): ✅ Jet Reversal Detected!"
                )
            else:
                st.error(
                    f"Latest Jet Check ({latest_jet['crossing_time']}): ❌ No Jet Reversal Detected."
                )

        # --- Jet Reversal Check Section ---
        st.header("1. Jet Reversal Parameters")

        with st.expander("Jet Reversal Advanced Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)
            dt = col1.number_input(
                "Time Window (dt) [s]",
                value=300,
                help=(
                    "Time window around the crossing time to search for jets (in seconds)."
                    if show_hints
                    else None
                ),
            )
            jet_len = col2.number_input(
                "Jet Length Threshold",
                value=3,
                help=(
                    "Minimum number of data points required to qualify as a valid jet."
                    if show_hints
                    else None
                ),
            )
            data_rate = col3.selectbox(
                "Data Rate",
                ["brst", "fast", "srvy"],
                index=0,
                help="Data rate resolution of the MMS instruments." if show_hints else None,
            )

            level = col1.selectbox(
                "Data Level",
                ["l2", "l1"],
                index=0,
                help=(
                    "Processing level of the MMS data (L2 is recommended for science analysis)."
                    if show_hints
                    else None
                ),
            )
            coord_type = col2.selectbox(
                "Coordinate Type",
                ["lmn", "gse"],
                index=0,
                help=(
                    "Coordinate system used for the magnetic field and velocity vectors."
                    if show_hints
                    else None
                ),
            )
            time_clip = col3.checkbox(
                "Time Clip",
                value=True,
                help=(
                    "Whether to strictly clip the data to the provided time window."
                    if show_hints
                    else None
                ),
            )

            t_delta = col1.number_input(
                "Find Next Jet Time Delta (mins)",
                value=10,
                help=(
                    "Time interval (in minutes) to step forward when searching for the next jet."
                    if show_hints
                    else None
                ),
            )
            max_attempts = col2.number_input(
                "Max Attempts for Next Jet",
                value=5,
                min_value=1,
                help=(
                    "Maximum number of time intervals to check before giving up."
                    if show_hints
                    else None
                ),
            )

        # --- Reconnection Location Section ---
        st.header("2. Reconnection Model Parameters")

        with st.expander("Reconnection Model Parameters", expanded=True):
            col1, col2 = st.columns(2)
            tsy_model = col1.selectbox(
                "Tsyganenko Model",
                ["t89", "t96", "t01", "t04s"],
                index=1,
                help=(
                    "Empirical magnetic field model used for the background magnetosphere."
                    if show_hints
                    else None
                ),
            )
            recon_models = col2.multiselect(
                "Reconnection Models",
                ["shear", "bisection", "reconnection energy", "exhaust velocity"],
                default=["shear", "bisection", "reconnection energy", "exhaust velocity"],
                help=(
                    "Physics models to calculate the probability of magnetic reconnection."
                    if show_hints
                    else None
                ),
            )

            omni_level = col1.selectbox(
                "OMNI Data Level",
                ["hro_1min", "hro_5min"],
                index=0,
                help=(
                    "Resolution of the OMNI solar wind data to pull for model parameters."
                    if show_hints
                    else None
                ),
            )
            m_p = col2.number_input(
                "Proton Mass Multiple (m_p)",
                value=1.0,
                help="Scaling factor for the proton mass in calculations." if show_hints else None,
            )
            dr = col1.number_input(
                "Grid Resolution (dr)",
                value=0.5,
                help=(
                    "Spatial resolution of the plotting grid in Earth Radii (RE)."
                    if show_hints
                    else None
                ),
            )
            limits = col2.number_input(
                "Grid Limits (±)",
                value=20.0,
                help=(
                    "Maximum extent of the X and Y plotting bounds (in RE)." if show_hints else None
                ),
            )

        def run_jet_check(c_time):
            """
            Executes the jet reversal detection algorithm for a specific crossing time.

            Args:
                c_time (datetime): The UTC crossing time to analyze.

            Returns:
                tuple: (bool success, dict data_dict) Returns True and the parsed parameters if a jet is found.
            """
            with st.spinner(
                f"Running Jet Reversal Check for MMS{mms_probe} at {c_time.strftime('%H:%M:%S')}..."
            ):
                try:
                    res = jet_reversal_check(
                        crossing_time=c_time,
                        dt=dt,
                        probe=mms_probe,
                        data_rate=data_rate,
                        level=level,
                        coord_type=coord_type,
                        time_clip=time_clip,
                        jet_len=jet_len,
                        date_obs=c_time.strftime("%Y%m%d"),
                        return_plotly_fig=True,
                        dark_mode=is_dark_mode,
                    )
                    if res is None:
                        st.error(
                            f"Magnetosphere or magnetosheath region not found for date {c_time.strftime('%Y-%m-%d %H:%M:%S+00:00')}"
                        )
                        return False, False

                    fig, jet_detection, data_dict = res

                    # Record history
                    run_record = {
                        "type": "jet_reversal",
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "crossing_time": c_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "mms_probe": mms_probe,
                        "dt": dt,
                        "jet_len": jet_len,
                        "data_rate": data_rate,
                        "level": level,
                        "coord_type": coord_type,
                        "time_clip": time_clip,
                        "jet_detection": jet_detection,
                        "fig": fig,
                    }
                    st.session_state["history"]["jet_checks"].append(run_record)
                    st.session_state["jet_view_idx"] = -1  # reset view to latest
                    save_auto_session()

                    return True, data_dict if jet_detection else None
                except Exception as e:
                    st.error(f"Error during Jet Reversal Check: {e}")
            return False, False

        def run_recon_models(c_time, save_data=False, csv_name="", det=None):
            """
            Computes and plots 3D reconnection locations based on selected models.

            Args:
                c_time (datetime): The UTC crossing time.
                save_data (bool): Whether to append the output to the stats CSV.
                csv_name (str): The filename for the output CSV.
                det (DataFrame): Existing jet detection parameters to use as background context.

            Returns:
                bool: True if successful, False otherwise.
            """
            if len(recon_models) == 0:
                st.error("Please select at least one reconnection model.")
                return False
            with st.spinner(
                f"Calculating Reconnection Models for MMS{mms_probe} at {c_time.strftime('%H:%M:%S')} (this may take a while)..."
            ):
                try:
                    model_mapping = {
                        "shear": {"var_idx": 3, "label": "Shear"},
                        "bisection": {"var_idx": 6, "label": "Bisection Field"},
                        "reconnection energy": {"var_idx": 4, "label": "Reconnection Energy"},
                        "exhaust velocity": {"var_idx": 5, "label": "Exhaust Velocity"},
                    }

                    trange_str = c_time.strftime("%Y-%m-%d %H:%M:%S")

                    model_inputs = {
                        "trange": [trange_str],
                        "probe": None,
                        "omni_level": omni_level.split("_")[0],
                        "mms_probe_num": str(mms_probe),
                        "model_type": tsy_model,
                        "m_p": m_p,
                        "dr": dr,
                        "min_max_val": limits,
                        "y_min": -limits,
                        "y_max": limits,
                        "z_min": -limits,
                        "z_max": limits,
                        "save_data": save_data,
                    }

                    import numpy as np

                    res = rx_model(**model_inputs)

                    if not res:
                        return False

                    images = []
                    c_labels = []

                    for model_name in recon_models:
                        var_idx = model_mapping[model_name]["var_idx"]
                        label = model_mapping[model_name]["label"]

                        raw_data = res[var_idx]
                        norm_data = (raw_data - np.nanmin(raw_data)) / (
                            np.nanmax(raw_data) - np.nanmin(raw_data)
                        )
                        images.append(norm_data)
                        c_labels.append(label)

                    sw_params = res[8]

                    figure_inputs = {
                        "image": images,
                        "convolution_order": [1] * len(images),
                        "t_range": [trange_str],
                        "b_imf": np.round(sw_params["b_imf"], 2),
                        "b_msh": np.round(sw_params["mms_b_gsm"], 2),
                        "xrange": [-limits, limits],
                        "yrange": [-limits, limits],
                        "mms_probe_num": str(mms_probe),
                        "mms_sc_pos": np.round(sw_params["mms_sc_pos"], 2),
                        "dr": dr,
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
                        "cmap_list": ["viridis", "cividis", "plasma", "magma"][: len(images)],
                        "draw_patch": [True] * len(images),
                        "draw_ridge": [True] * len(images),
                        "save_fig": False,
                        "fig_name": "rxn_app_figure",
                        "fig_format": "html",
                        "c_label": c_labels,
                        "wspace": 0.15,
                        "hspace": 0.17,
                        "fig_size": (8.775, 10),
                        "box_style": dict(boxstyle="round", color="k", alpha=0.8),
                        "title_y_pos": 1.09,
                        "interpolation": "None",
                        "tsy_model": tsy_model,
                        "dark_mode": is_dark_mode,
                        "save_rc_file": save_data,
                        "rc_file_name": csv_name,
                        "rc_folder": "./",
                        "df_jet_reversal": det,
                    }

                    rx_fig = ridge_finder_multiple_interactive(**figure_inputs)

                    # Record history
                    run_record = {
                        "type": "recon_models",
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "crossing_time": c_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "mms_probe": mms_probe,
                        "tsy_model": tsy_model,
                        "recon_models": recon_models,
                        "omni_level": omni_level,
                        "m_p": m_p,
                        "dr": dr,
                        "limits": limits,
                        "fig": rx_fig,
                    }
                    st.session_state["history"]["recon_models"].append(run_record)
                    st.session_state["rxn_view_idx"] = -1
                    save_auto_session()

                    return True
                except Exception as e:
                    st.error(f"Error running Reconnection Models: {e}")
            return False

    with sidebar_actions:
        st.header("Actions")
        if st.button("Run Jet Reversal Check", width="stretch"):
            if crossing_time:
                success, jet_det = run_jet_check(crossing_time)
                if success:
                    st.success("Jet Reversal check completed!")
            else:
                st.error("Invalid crossing time.")

        if st.button("Find Next Jet", width="stretch"):
            if crossing_time:
                current_test_time = crossing_time
                found = False
                for attempt in range(max_attempts):
                    current_test_time = current_test_time + datetime.timedelta(minutes=t_delta)

                    success, jet_det = run_jet_check(current_test_time)
                    st.session_state.crossing_time_str = current_test_time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    if success and jet_det:
                        st.success(
                            f"Jet found at {current_test_time.strftime('%Y-%m-%d %H:%M:%S')}!"
                        )
                        found = True
                        break

                if not found:
                    st.warning(
                        f"No jet found after {max_attempts} attempts. Stopped at {current_test_time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                save_auto_session()
                st.rerun()
            else:
                st.error("Invalid crossing time.")

        if st.button("Run Reconnection Models", width="stretch"):
            if crossing_time:
                success = run_recon_models(crossing_time)
                if success:
                    st.success("Reconnection Models generated!")
            else:
                st.error("Invalid crossing time.")

        if st.button("Run All", type="primary", width="stretch"):
            if crossing_time:
                s1, det = run_jet_check(crossing_time)
                s2 = run_recon_models(crossing_time, det=det)
                if s1 and s2:
                    st.success("All checks and models generated successfully!")
            else:
                st.error("Invalid crossing time.")

        if st.button("Run Statistics Mode", width="stretch", disabled=not use_stats_mode):
            if not use_stats_mode:
                st.warning("Please enable Statistics Mode in the sidebar first.")
            elif crossing_time:
                if (
                    stop_condition == "End Time"
                    and stats_end_time
                    and crossing_time > stats_end_time
                ):
                    st.error("Invalid time range. Start time must be before end time.")
                else:
                    curr_t = crossing_time
                    t_delta = datetime.timedelta(minutes=stats_delta_mins)

                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    if stop_condition == "End Time":
                        total_steps = (
                            int(
                                (stats_end_time - crossing_time).total_seconds()
                                // (stats_delta_mins * 60)
                            )
                            + 1
                        )
                    else:
                        total_steps = target_jets

                    step = 0
                    jets_found = 0

                    while True:
                        if stop_condition == "End Time" and curr_t > stats_end_time:
                            break
                        if stop_condition == "Target Number of Jets" and jets_found >= target_jets:
                            break

                        # fail-safe for target jets to avoid infinite loop
                        if stop_condition == "Target Number of Jets" and step > target_jets * 100:
                            st.warning(
                                "Reached maximum search attempts (100x target). Stopping early."
                            )
                            break

                        if stop_condition == "End Time":
                            status_text.text(
                                f"Processing {curr_t.strftime('%Y-%m-%d %H:%M:%S')} (Step {step+1}/{total_steps})"
                            )
                            top_counter_placeholder.markdown(
                                f"<h1 style='text-align: center; color: #4CAF50;'>Jets Found: {jets_found}</h1>",
                                unsafe_allow_html=True,
                            )
                        else:
                            status_text.text(
                                f"Processing {curr_t.strftime('%Y-%m-%d %H:%M:%S')} (Found {jets_found}/{target_jets})"
                            )
                            top_counter_placeholder.markdown(
                                f"<h1 style='text-align: center; color: #4CAF50;'>Jets Found: {jets_found} / {target_jets}</h1>",
                                unsafe_allow_html=True,
                            )

                        s1, det = run_jet_check(curr_t)

                        if len(st.session_state["history"]["jet_checks"]) > 0:
                            latest_jet_fig = st.session_state["history"]["jet_checks"][-1].get(
                                "fig"
                            )
                            if latest_jet_fig:
                                with live_jet_plot_placeholder.container():
                                    st.plotly_chart(
                                        latest_jet_fig,
                                        use_container_width=True,
                                        key=f"live_jet_{step}",
                                    )

                        if s1 and det:
                            run_recon_models(
                                curr_t, save_data=True, csv_name=stats_csv_name, det=det
                            )
                            jets_found += 1

                        step += 1
                        if stop_condition == "End Time":
                            progress_bar.progress(min(step / total_steps, 1.0))
                        else:
                            progress_bar.progress(min(jets_found / target_jets, 1.0))

                        curr_t += t_delta

                    status_text.text("Statistics Mode Completed!")
                    if stop_condition == "End Time":
                        top_counter_placeholder.markdown(
                            f"<h1 style='text-align: center; color: #4CAF50;'>Statistics Mode Completed! Total Jets Found: {jets_found}</h1>",
                            unsafe_allow_html=True,
                        )
                    else:
                        top_counter_placeholder.markdown(
                            f"<h1 style='text-align: center; color: #4CAF50;'>Statistics Mode Completed! Found {jets_found} / {target_jets} Jets.</h1>",
                            unsafe_allow_html=True,
                        )
                    st.session_state["latest_stats_csv"] = stats_csv_name
            else:
                st.error("Invalid start time.")

        if st.button("Reset All", type="secondary", width="stretch"):
            reset_session()
            st.rerun()

    def get_dropdown_options(history_list):
        options = []
        for i, run in enumerate(history_list):
            ct = run["crossing_time"]
            ts = run["timestamp"]
            options.append(f"Run {i+1} | Time: {ct} (Tested at {ts})")
        return options

    with tab_jet:
        st.header("Jet Reversal Results")
        jet_hist = st.session_state["history"]["jet_checks"]

        if len(jet_hist) > 0:
            options = get_dropdown_options(jet_hist)
            # Handle current index
            curr_idx = st.session_state["jet_view_idx"]
            if curr_idx < 0:
                curr_idx = len(jet_hist) - 1
            if curr_idx >= len(jet_hist):
                curr_idx = len(jet_hist) - 1

            col_sel, col_prev, col_next = st.columns([6, 2, 2])
            selected_option = col_sel.selectbox("Browse Run History", options, index=curr_idx)
            new_idx = options.index(selected_option)

            if col_prev.button("Previous Run", key="prev_jet", width="stretch") and curr_idx > 0:
                new_idx = curr_idx - 1
            if (
                col_next.button("Next Run", key="next_jet", width="stretch")
                and curr_idx < len(jet_hist) - 1
            ):
                new_idx = curr_idx + 1

            if new_idx != curr_idx:
                st.session_state["jet_view_idx"] = new_idx
                st.rerun()

            active_run = jet_hist[new_idx]

            if active_run["jet_detection"]:
                st.success("✅ Jet Reversal Detected!")
            else:
                st.error("❌ No Jet Reversal Detected.")

            st.write(
                f"**Parameters used:** Probe {active_run['mms_probe']}, dt {active_run['dt']}, jet len {active_run['jet_len']}, rate {active_run['data_rate']}, level {active_run['level']}, type {active_run['coord_type']}, clip {active_run['time_clip']}"
            )

            if active_run["fig"]:
                st.plotly_chart(active_run["fig"], width="stretch")
        else:
            st.info("Run the Jet Reversal Check from the Actions sidebar first.")

    with tab_rxn:
        st.header("Reconnection Model Results")
        rxn_hist = st.session_state["history"]["recon_models"]

        if len(rxn_hist) > 0:
            options = get_dropdown_options(rxn_hist)
            # Handle current index
            curr_idx = st.session_state["rxn_view_idx"]
            if curr_idx < 0:
                curr_idx = len(rxn_hist) - 1
            if curr_idx >= len(rxn_hist):
                curr_idx = len(rxn_hist) - 1

            col_sel, col_prev, col_next = st.columns([6, 2, 2])
            selected_option = col_sel.selectbox(
                "Browse Run History", options, index=curr_idx, key="sel_rxn"
            )
            new_idx = options.index(selected_option)

            if col_prev.button("Previous Run", key="prev_rxn", width="stretch") and curr_idx > 0:
                new_idx = curr_idx - 1
            if (
                col_next.button("Next Run", key="next_rxn", width="stretch")
                and curr_idx < len(rxn_hist) - 1
            ):
                new_idx = curr_idx + 1

            if new_idx != curr_idx:
                st.session_state["rxn_view_idx"] = new_idx
                st.rerun()

            active_run = rxn_hist[new_idx]

            # Find matching jet detection if available
            matching_jet = next(
                (
                    j
                    for j in st.session_state["history"]["jet_checks"]
                    if j["crossing_time"] == active_run["crossing_time"]
                ),
                None,
            )
            if matching_jet is not None:
                if matching_jet["jet_detection"]:
                    st.success("✅ Jet Reversal was detected at this time!")
                else:
                    st.error("❌ No Jet Reversal was detected at this time.")
            else:
                st.warning("⚠️ Jet Reversal Check has not been run for this time yet.")

            st.write(
                f"**Parameters used:** Probe {active_run['mms_probe']}, Model {active_run['tsy_model']}, m_p {active_run['m_p']}, dr {active_run['dr']}, limit ±{active_run['limits']}, OMNI {active_run['omni_level']}"
            )
            st.write(f"**Recon Models:** {', '.join(active_run['recon_models'])}")

            if active_run["fig"]:
                st.plotly_chart(active_run["fig"], width="stretch")
        else:
            st.info("Run the Reconnection Models from the Actions sidebar first.")

    if tab_stats is not None:
        with tab_stats:
            st.header("Statistics Results")

            st.markdown("Generate statistical plots from batch runs or uploaded CSV data.")

            # CSV handling
            default_csv = st.session_state.get("latest_stats_csv", "")
            csv_source = st.radio(
                "CSV Source", ["Use generated from Statistics Mode", "Upload existing CSV"]
            )

            csv_path = None
            if csv_source == "Use generated from Statistics Mode":
                if os.path.exists(default_csv):
                    csv_path = default_csv
                    st.success(f"Using `{default_csv}`")
                    with open(csv_path, "rb") as f:
                        st.download_button(
                            "Download CSV", data=f, file_name=default_csv, mime="text/csv"
                        )
                else:
                    st.warning("No generated CSV found. Run Statistics Mode first.")
            else:
                uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
                if uploaded_csv is not None:
                    csv_path = "temp_uploaded_stats.csv"
                    with open(csv_path, "wb") as f:
                        f.write(uploaded_csv.getbuffer())
                    st.success("CSV Uploaded successfully.")
            if csv_path and os.path.exists(csv_path):
                st.success(f"Using statistics data from: {csv_path}")

                # Plot selection
                var_options = {
                    "b_imf_z": r"IMF Bz",
                    "b_imf_x": r"IMF Bx",
                    "b_imf_y": r"IMF By",
                    "imf_clock_angle": r"IMF Clock Angle",
                    "cone_angle": r"Cone Angle",
                    "p_dyn": r"Dynamic Pressure",
                    "msh_msp_shear": r"Shear Angle",
                    "r_rc": "Reconnection Distance",
                    "delta_beta": "Delta Beta",
                }
                plots_to_gen = st.multiselect(
                    "Select Figures to Generate",
                    [
                        "Histograms",
                        "KDE Plots",
                        "2D Histograms",
                        "Scatter Plots",
                        "MMS Location Scatter Plot",
                    ],
                    default=["Histograms", "KDE Plots", "2D Histograms", "Scatter Plots"],
                )

                x_vars = st.multiselect(
                    "X-Axis Variable(s)",
                    list(var_options.keys()),
                    format_func=lambda x: var_options[x],
                    default=["b_imf_z"],
                )
                y_vars = st.multiselect(
                    "Y-Axis Variable(s)",
                    list(var_options.keys()),
                    format_func=lambda x: var_options[x],
                    default=["r_rc"],
                )

                if st.button("Generate Selected Figures"):
                    with st.spinner("Generating statistical figures..."):
                        if not x_vars or not y_vars:
                            st.error("Please select at least one X and one Y variable.")
                        elif len(x_vars) != len(y_vars):
                            st.error(
                                f"Please select the same number of X and Y variables for pairwise plotting (you selected {len(x_vars)} X and {len(y_vars)} Y)."
                            )
                        else:
                            for x_var, y_var in zip(x_vars, y_vars):
                                if len(x_vars) > 1:
                                    st.markdown(f"### {var_options[x_var]} vs {var_options[y_var]}")

                                figures, err = generate_statistics_plots(
                                    csv_path,
                                    dark_mode=is_dark_mode,
                                    selected_plots=plots_to_gen,
                                    x_var=x_var,
                                    y_var=y_var,
                                )
                                if err:
                                    st.error(err)
                                else:
                                    for title, fig in figures.items():
                                        if len(x_vars) == 1:
                                            st.write(f"### {title}")
                                        else:
                                            st.write(f"**{title}**")

                                        if fig:
                                            st.pyplot(fig)
                                        else:
                                            st.warning(f"Could not generate {title}")


if __name__ == "__main__":
    main()
