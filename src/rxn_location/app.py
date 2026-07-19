"""
Streamlit GUI Application for Reconnection Location Analysis

This module contains the interactive web interface for the `rxn_location` package.
It integrates Jet Reversal detection, 3D Reconnection model visualization,
and batch processing Statistics Mode for analyzing MMS spacecraft data.

Features include:
- Master Jet List with persistent JSON storage and 2-minute deduplication
- Interactive data table with sorting, manual pruning, and export
- Save/Load parameter presets
- Dynamic filtering on statistics plots
- Local data cache management dashboard
- Quick re-run from master list entries
"""

import streamlit as st
import streamlit.components.v1 as components
import datetime
import plotly.io as pio
import pandas as pd
import io
import json
import shutil
import zipfile
import pytz
import pickle
import os
import logging
from pathlib import Path

# Try importing the required project modules
try:
    from rxn_location.jet_reversal_check_function import jet_reversal_check
    from rxn_location.rx_model_funcs import rx_model, ridge_finder_multiple_interactive
    from rxn_location.app_stats_plots_interactive import generate_interactive_plots
    from rxn_location.app_seaborn_plots import generate_seaborn_jointplots
    from rxn_location import master_jet_list as mjl
    from rxn_location import presets as preset_mgr
except ImportError as e:
    st.error(
        f"Error importing rxn_location modules. Ensure the package is installed properly.\n{e}"
    )

# Page config
st.set_page_config(
    page_title="RXN Location Dash",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "show_toast" in st.session_state:
    st.toast(st.session_state.pop("show_toast"))

AUTO_SAVE_PATH = Path(os.path.expanduser("~")) / ".rxn_location_auto_save.pkl"


def save_auto_session():
    """
    Saves the current application state to a local session file.
    
    This ensures that user inputs, such as active time, selected probe, and 
    data rates are preserved across page reloads.
    """
    state_to_save = {
        "history": st.session_state.get(
            "history", {"jet_checks": [], "recon_models": []}
        ),
        "crossing_time_str": st.session_state.get(
            "crossing_time_str", "2015-09-02 16:45:00"
        ),
        "dark_mode": st.session_state.get("dark_mode", True),
    }
    try:
        with open(AUTO_SAVE_PATH, "wb") as f:
            pickle.dump(state_to_save, f)
    except Exception as e:
        pass  # silently fail on auto-save errors


def load_auto_session():
    """
    Loads a previously saved application state from a local session file 
    into the current Streamlit session.
    """
    if AUTO_SAVE_PATH.exists():
        try:
            with open(AUTO_SAVE_PATH, "rb") as f:
                saved_state = pickle.load(f)
            return saved_state
        except:
            return None
    return None


def reset_session():
    """
    Resets the application session state to default values.
    """
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
        "duplicate_dialog_state",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]


def inject_beforeunload():
    """
    Injects JavaScript into the Streamlit app to trigger a callback when the user closes the window.
    """
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


def _get_pyspedas_data_dir():
    """
    Determine the pyspedas data directory by checking config, then falling back
    to ~/pyspedas_data/.

    Returns
    -------
    Path or None
        The pyspedas data directory path, or None if it doesn't exist.
    """
    # Try to get from pyspedas config
    try:
        import pyspedas

        config = pyspedas.config
        if hasattr(config, "CONFIG_FILE"):
            # Try reading the config file
            import configparser

            cp = configparser.ConfigParser()
            cp.read(config.CONFIG_FILE)
            data_dir = cp.get("pyspedas", "local_data_dir", fallback=None)
            if data_dir:
                p = Path(os.path.expanduser(data_dir))
                if p.exists():
                    return p
    except Exception:
        pass

    # Fall back to default locations
    for candidate in [
        Path(os.path.expanduser("~")) / "pyspedas_data",
        Path(os.path.expanduser("~")) / ".pyspedas",
    ]:
        if candidate.exists():
            return candidate

    return None


def _get_dir_size_info(directory):
    """
    Calculate size and file count for a directory.

    Parameters
    ----------
    directory : Path
        The directory to scan.

    Returns
    -------
    tuple of (int, int, datetime or None)
        (total_bytes, file_count, oldest_file_mtime)
    """
    total_size = 0
    file_count = 0
    oldest_mtime = None

    for root, dirs, files in os.walk(directory):
        for f in files:
            fp = Path(root) / f
            try:
                stat = fp.stat()
                total_size += stat.st_size
                file_count += 1
                mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
                if oldest_mtime is None or mtime < oldest_mtime:
                    oldest_mtime = mtime
            except OSError:
                continue

    return total_size, file_count, oldest_mtime


def _format_size(size_bytes):
    """Format bytes into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def _clear_old_files(directory, days=30):
    """
    Delete files older than a given number of days from a directory.

    Parameters
    ----------
    directory : Path
        The directory to clean.
    days : int
        Files older than this many days will be deleted.

    Returns
    -------
    int
        Number of files deleted.
    """
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    deleted = 0
    for root, dirs, files in os.walk(directory, topdown=False):
        for f in files:
            fp = Path(root) / f
            try:
                mtime = datetime.datetime.fromtimestamp(fp.stat().st_mtime)
                if mtime < cutoff:
                    fp.unlink()
                    deleted += 1
            except OSError:
                continue
        # Remove empty directories
        try:
            rd = Path(root)
            if rd != directory and not any(rd.iterdir()):
                rd.rmdir()
        except OSError:
            continue
    return deleted


def main():
    """
    The main entry point for the Streamlit application layout and logic.
    """
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("RXN Location Analyzer")
        st.markdown("Analyze Jet Reversals and Reconnection Models using MMS data.")
    with col2:
        top_counter_placeholder = st.empty()

    inject_beforeunload()

    # Initialization — always start fresh (users can restore via Session Manager)
    if "initialized" not in st.session_state:
        st.session_state["history"] = {"jet_checks": [], "recon_models": []}
        st.session_state["crossing_time_str"] = "2015-09-02 16:45:00"
        st.session_state["dark_mode"] = True

        st.session_state["initialized"] = True
        st.session_state["jet_view_idx"] = -1
        st.session_state["rxn_view_idx"] = -1

    # Load master jet list into session state
    if "master_jets" not in st.session_state:
        st.session_state["master_jets"] = mjl.load_master_list()

    # Initialize duplicate dialog state
    if "duplicate_dialog_state" not in st.session_state:
        st.session_state["duplicate_dialog_state"] = None

    # =========================================================================
    # SIDEBAR
    # =========================================================================

    # --- Session Manager ---
    with st.sidebar.expander("Session Manager", expanded=False):
        uploaded_file = st.file_uploader(
            "Restore Session (.rxn_session)", type=["rxn_session", "pkl"]
        )
        if uploaded_file is not None:
            if st.button("Load Session", width="stretch"):
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
                width="stretch",
            )
        except Exception as e:
            st.error("Cannot serialize session.")

    # --- Presets Manager (Feature #10) ---
    with st.sidebar.expander("Parameter Presets", expanded=False):
        presets = preset_mgr.load_presets()
        preset_names = sorted(presets.keys())

        if preset_names:
            selected_preset = st.selectbox(
                "Load a Preset", ["(none)"] + preset_names, key="preset_selector"
            )
            col_load, col_del = st.columns(2)
            if col_load.button("Load Preset", width="stretch"):
                if selected_preset and selected_preset != "(none)":
                    params = presets[selected_preset]
                    st.session_state["crossing_time_str"] = params.get(
                        "crossing_time_str", st.session_state["crossing_time_str"]
                    )
                    # Store preset values in session state for sidebar widgets to pick up
                    for key in [
                        "mms_probe",
                        "dt",
                        "jet_len",
                        "data_rate",
                        "level",
                        "coord_type",
                        "time_clip",
                        "t_delta",
                        "max_attempts",
                        "tsy_model",
                        "recon_models",
                        "omni_level",
                        "m_p",
                        "dr",
                        "limits",
                    ]:
                        if key in params:
                            st.session_state[f"preset_{key}"] = params[key]
                    save_auto_session()
                    st.success(f"Preset '{selected_preset}' loaded!")
                    st.rerun()

            if col_del.button("Delete Preset", width="stretch"):
                if selected_preset and selected_preset != "(none)":
                    preset_mgr.delete_preset(selected_preset)
                    st.success(f"Preset '{selected_preset}' deleted.")
                    st.rerun()
        else:
            st.info("No presets saved yet.")

        st.markdown("---")
        new_preset_name = st.text_input("New Preset Name", key="new_preset_name")
        if st.button("Save Current Settings as Preset", width="stretch"):
            if new_preset_name.strip():
                # Collect current params from session state
                current_params = {
                    "crossing_time_str": st.session_state.get(
                        "crossing_time_str", "2015-09-02 16:45:00"
                    ),
                }
                # These will be read from the widget values via session state keys
                for key in [
                    "mms_probe",
                    "dt",
                    "jet_len",
                    "data_rate",
                    "level",
                    "coord_type",
                    "time_clip",
                    "t_delta",
                    "max_attempts",
                    "tsy_model",
                    "recon_models",
                    "omni_level",
                    "m_p",
                    "dr",
                    "limits",
                ]:
                    if f"preset_{key}" in st.session_state:
                        current_params[key] = st.session_state[f"preset_{key}"]
                preset_mgr.save_preset(new_preset_name.strip(), current_params)
                st.success(f"Preset '{new_preset_name.strip()}' saved!")
                st.rerun()
            else:
                st.warning("Please enter a preset name.")

    # --- Global Settings ---
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

    verbosity = st.sidebar.selectbox(
        "Verbosity Level",
        [0, 1, 2, 3],
        index=[0, 1, 2, 3].index(st.session_state.get("preset_verbosity", 2)),
        help=(
            "0: Silent, 1: Important Only, 2: Standard (No PySPEDAS), 3: All. Default: 2"
            if show_hints
            else None
        ),
    )
    st.session_state["preset_verbosity"] = verbosity
    from rxn_location.logger import set_verbosity

    set_verbosity(verbosity)

    # --- Observation Parameters ---
    st.sidebar.header("Observation Parameters")

    date_str = st.sidebar.text_input(
        "Crossing Time (UTC)", value=st.session_state.crossing_time_str
    )
    try:
        from dateutil import parser
        import pytz

        crossing_time = parser.parse(date_str)
        if crossing_time.tzinfo is None:
            crossing_time = crossing_time.replace(tzinfo=pytz.utc)
        # Normalize the string back to standard format
        clean_str = crossing_time.strftime("%Y-%m-%d %H:%M:%S")
        if date_str != clean_str:
            st.session_state.crossing_time_str = clean_str
            st.rerun()
        st.session_state.crossing_time_str = clean_str
    except Exception as e:
        st.sidebar.error("Invalid time format. Try YYYY-MM-DD HH:MM:SS")
        crossing_time = None

    mms_probe = st.sidebar.selectbox(
        "MMS Probe",
        [1, 2, 3, 4],
        index=(
            st.session_state.get("preset_mms_probe", 3) - 1
            if st.session_state.get("preset_mms_probe") in [1, 2, 3, 4]
            else 2
        ),
        help=(
            "Select which MMS spacecraft probe to use for the observation data."
            if show_hints
            else None
        ),
    )
    st.session_state["preset_mms_probe"] = mms_probe

    st.sidebar.markdown("---")
    st.sidebar.header("Statistics Mode Parameters")
    use_stats_mode = st.sidebar.checkbox("Enable Statistics Mode")
    stop_condition = st.sidebar.radio(
        "Stop Condition",
        ["End Time", "Target Number of Jets", "From File List"],
        disabled=not use_stats_mode,
    )

    stats_uploaded_file = None
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
    elif stop_condition == "Target Number of Jets":
        target_jets = st.sidebar.number_input(
            "Number of Jets to Find", min_value=1, value=10, disabled=not use_stats_mode
        )
        stats_end_time = None
        end_str_safe = f"{target_jets}jets"
    else:
        stats_uploaded_file = st.sidebar.file_uploader(
            "Upload Time List (.csv, .txt)",
            type=["csv", "txt"],
            disabled=not use_stats_mode,
            help=(
                "Each line or row should contain a parseable time string."
                if show_hints
                else None
            ),
        )
        target_jets = None
        stats_end_time = None
        end_str_safe = "filelist"

    stats_delta_mins = st.sidebar.number_input(
        "Time Delta (minutes)",
        min_value=1,
        value=15,
        help="Step size for the batch loop." if show_hints else None,
        disabled=(not use_stats_mode) or (stop_condition == "From File List"),
    )
    start_str_safe = date_str.replace(" ", "_").replace(":", "")
    default_csv_name = (
        f"reconnection_stats_{start_str_safe}_{end_str_safe}_{stats_delta_mins}m.csv"
    )
    stats_csv_name = st.sidebar.text_input(
        "Output CSV Name",
        value=default_csv_name,
        help="Name of the file to save batch data to." if show_hints else None,
        disabled=not use_stats_mode,
    )

    # --- Cache Dashboard (Feature #12) ---
    with st.sidebar.expander("Data Cache Management", expanded=False):
        cache_dir = _get_pyspedas_data_dir()
        if cache_dir is not None and cache_dir.exists():
            total_size, file_count, oldest_mtime = _get_dir_size_info(cache_dir)
            st.metric("Cache Size", _format_size(total_size))
            st.metric("Files", f"{file_count:,}")
            if oldest_mtime:
                st.caption(f"Oldest file: {oldest_mtime.strftime('%Y-%m-%d')}")
            st.caption(f"Location: `{cache_dir}`")

            col_clear1, col_clear2 = st.columns(2)
            if col_clear1.button("Clear All", width="stretch", key="cache_clear_all"):
                try:
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    st.success("Cache cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing cache: {e}")

            if col_clear2.button(
                "Clear > 30 Days", width="stretch", key="cache_clear_old"
            ):
                deleted = _clear_old_files(cache_dir, days=30)
                st.success(f"Deleted {deleted} files older than 30 days.")
                st.rerun()
        else:
            st.info(
                "No pyspedas data cache found. Data will be cached in "
                "`~/pyspedas_data/` after the first run."
            )

    sidebar_actions = st.sidebar.container()  # placeholder for buttons

    # =========================================================================
    # TABS
    # =========================================================================

    if use_stats_mode:
        tab_controls, tab_jet, tab_rxn, tab_master, tab_stats = st.tabs(
            [
                "Controls",
                "Jet Reversal Plot",
                "Reconnection Models",
                "Master Jet List",
                "Statistics Results",
            ]
        )
    else:
        tab_controls, tab_jet, tab_rxn, tab_master = st.tabs(
            ["Controls", "Jet Reversal Plot", "Reconnection Models", "Master Jet List"]
        )
        tab_stats = None

    with tab_jet:
        live_jet_plot_placeholder = st.empty()

    # =========================================================================
    # CONTROLS TAB
    # =========================================================================

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

            _default_dt = st.session_state.get("preset_dt", 300)
            dt = col1.number_input(
                "Time Window (dt) [s]",
                value=_default_dt,
                help=(
                    "Time window around the crossing time to search for jets (in seconds)."
                    if show_hints
                    else None
                ),
            )
            st.session_state["preset_dt"] = dt

            _default_jet_len = st.session_state.get("preset_jet_len", 3)
            jet_len = col2.number_input(
                "Jet Length Threshold",
                value=_default_jet_len,
                help=(
                    "Minimum number of data points required to qualify as a valid jet."
                    if show_hints
                    else None
                ),
            )
            st.session_state["preset_jet_len"] = jet_len

            _dr_options = ["brst", "fast", "srvy"]
            _default_data_rate = st.session_state.get("preset_data_rate", "brst")
            _dr_index = (
                _dr_options.index(_default_data_rate)
                if _default_data_rate in _dr_options
                else 0
            )
            data_rate = col3.selectbox(
                "Data Rate",
                _dr_options,
                index=_dr_index,
                help=(
                    "Data rate resolution of the MMS instruments."
                    if show_hints
                    else None
                ),
            )
            st.session_state["preset_data_rate"] = data_rate

            _level_options = ["l2", "l1"]
            _default_level = st.session_state.get("preset_level", "l2")
            _lv_index = (
                _level_options.index(_default_level)
                if _default_level in _level_options
                else 0
            )
            level = col1.selectbox(
                "Data Level",
                _level_options,
                index=_lv_index,
                help=(
                    "Processing level of the MMS data (L2 is recommended for science analysis)."
                    if show_hints
                    else None
                ),
            )
            st.session_state["preset_level"] = level

            _coord_options = ["lmn", "gse"]
            _default_coord = st.session_state.get("preset_coord_type", "lmn")
            _co_index = (
                _coord_options.index(_default_coord)
                if _default_coord in _coord_options
                else 0
            )
            coord_type = col2.selectbox(
                "Coordinate Type",
                _coord_options,
                index=_co_index,
                help=(
                    "Coordinate system used for the magnetic field and velocity vectors."
                    if show_hints
                    else None
                ),
            )
            st.session_state["preset_coord_type"] = coord_type

            _default_time_clip = st.session_state.get("preset_time_clip", True)
            time_clip = col3.checkbox(
                "Time Clip",
                value=_default_time_clip,
                help=(
                    "Whether to strictly clip the data to the provided time window."
                    if show_hints
                    else None
                ),
            )
            st.session_state["preset_time_clip"] = time_clip

            _default_t_delta = st.session_state.get("preset_t_delta", 10)
            t_delta = col1.number_input(
                "Find Next Jet Time Delta (mins)",
                value=_default_t_delta,
                help=(
                    "Time interval (in minutes) to step forward when searching for the next jet."
                    if show_hints
                    else None
                ),
            )
            st.session_state["preset_t_delta"] = t_delta

            _default_max_attempts = st.session_state.get("preset_max_attempts", 5)
            max_attempts = col2.number_input(
                "Max Attempts for Next Jet",
                value=_default_max_attempts,
                min_value=1,
                help=(
                    "Maximum number of time intervals to check before giving up."
                    if show_hints
                    else None
                ),
            )
            st.session_state["preset_max_attempts"] = max_attempts

        # --- Reconnection Location Section ---
        st.header("2. Reconnection Model Parameters")

        with st.expander("Reconnection Model Parameters", expanded=True):
            col1, col2 = st.columns(2)

            _tsy_options = ["t89", "t96", "t01", "t04s"]
            _default_tsy = st.session_state.get("preset_tsy_model", "t96")
            _tsy_index = (
                _tsy_options.index(_default_tsy) if _default_tsy in _tsy_options else 1
            )
            tsy_model = col1.selectbox(
                "Tsyganenko Model",
                _tsy_options,
                index=_tsy_index,
                help=(
                    "Empirical magnetic field model used for the background magnetosphere."
                    if show_hints
                    else None
                ),
            )
            st.session_state["preset_tsy_model"] = tsy_model

            _default_recon_models = st.session_state.get(
                "preset_recon_models",
                ["shear", "reconnection energy", "exhaust velocity", "bisection"],
            )
            recon_models = col2.multiselect(
                "Reconnection Models",
                ["shear", "reconnection energy", "exhaust velocity", "bisection"],
                default=_default_recon_models,
                help=(
                    "Physics models to calculate the probability of magnetic reconnection."
                    if show_hints
                    else None
                ),
            )
            st.session_state["preset_recon_models"] = recon_models
            st.markdown("##### Plotting Options")
            # Removed the checkbox layout since there's only one model now

            _omni_options = ["hro_1min", "hro_5min"]
            _default_omni = st.session_state.get("preset_omni_level", "hro_1min")
            _omni_index = (
                _omni_options.index(_default_omni)
                if _default_omni in _omni_options
                else 0
            )
            omni_level = col1.selectbox(
                "OMNI Data Level",
                _omni_options,
                index=_omni_index,
                help=(
                    "Resolution of the OMNI solar wind data to pull for model parameters."
                    if show_hints
                    else None
                ),
            )
            st.session_state["preset_omni_level"] = omni_level

            _default_m_p = st.session_state.get("preset_m_p", 1.0)
            m_p = col2.number_input(
                "Proton Mass Multiple (m_p)",
                value=_default_m_p,
                help=(
                    "Scaling factor for the proton mass in calculations."
                    if show_hints
                    else None
                ),
            )
            st.session_state["preset_m_p"] = m_p

            _default_dr = st.session_state.get("preset_dr", 0.5)
            dr = col1.number_input(
                "Grid Resolution (dr)",
                value=_default_dr,
                help=(
                    "Spatial resolution of the plotting grid in Earth Radii (RE)."
                    if show_hints
                    else None
                ),
            )
            st.session_state["preset_dr"] = dr

            _default_limits = st.session_state.get("preset_limits", 20.0)
            limits = col2.number_input(
                "Grid Limits (±)",
                value=_default_limits,
                help=(
                    "Maximum extent of the X and Y plotting bounds (in RE)."
                    if show_hints
                    else None
                ),
            )
            st.session_state["preset_limits"] = limits

        # ==================================================================
        # Core analysis functions
        # ==================================================================

        jet_error_placeholder = st.empty()

        def _get_current_run_params():
            """Collect the current sidebar parameters into a dict."""
            return {
                "mms_probe": mms_probe,
                "dt": dt,
                "jet_len": jet_len,
                "data_rate": data_rate,
                "level": level,
                "coord_type": coord_type,
                "time_clip": time_clip,
                "t_delta": t_delta,
                "max_attempts": max_attempts,
                "tsy_model": tsy_model,
                "recon_models": recon_models,
                "omni_level": omni_level,
                "m_p": m_p,
                "dr": dr,
                "limits": limits,
            }

        def run_jet_check(c_time, skip_master_add=False):
            """
            Executes the jet reversal detection algorithm for a specific crossing time.

            Args:
                c_time (datetime): The UTC crossing time to analyze.
                skip_master_add (bool): If True, skip adding to the master list.

            Returns:
                tuple: (bool success, dict data_dict) Returns True and the parsed
                       parameters if a jet is found.
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
                        jet_error_placeholder.error(
                            f"Magnetosphere or magnetosheath region not found for date {c_time.strftime('%Y-%m-%d %H:%M:%S+00:00')}"
                        )
                        return False, False
                    else:
                        jet_error_placeholder.empty()

                    fig, jet_detection, data_dict = res

                    # Record history
                    run_record = {
                        "type": "jet_reversal",
                        "timestamp": datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
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
                    if "sel_jet" in st.session_state:
                        del st.session_state["sel_jet"]
                    save_auto_session()

                    # --- Master List Integration (Features #1-#4) ---
                    sw_params = None
                    if jet_detection and data_dict and not skip_master_add:
                        # Compute R_rc automatically in the background before saving (Option B)
                        success_rm, dist_rc_dict, sw_params = run_recon_models(
                            c_time, save_data=False, det=data_dict
                        )
                        if success_rm and dist_rc_dict:
                            data_dict.update(dist_rc_dict)

                        # Fallback to fetch SW params directly if recon models were skipped
                        if not sw_params:
                            try:
                                from rx_model_funcs import get_sw_params

                                trange_date_min = c_time - datetime.timedelta(
                                    minutes=30
                                )
                                trange_date_max = c_time + datetime.timedelta(
                                    minutes=30
                                )
                                trange_min_str = (
                                    trange_date_min.strftime("%Y-%m-%d %H:%M:%S") + "Z"
                                )
                                trange_max_str = (
                                    trange_date_max.strftime("%Y-%m-%d %H:%M:%S") + "Z"
                                )
                                omni_lvl = st.session_state.get(
                                    "omni_level", "hro_1min"
                                ).split("_")[0]
                                probe_num = str(st.session_state.get("mms_probe", 3))
                                sw_params = get_sw_params(
                                    trange=[trange_min_str, trange_max_str],
                                    omni_level=omni_lvl,
                                    mms_probe_num=probe_num,
                                )
                            except Exception as e:
                                print(f"Error fetching OMNI SW params: {e}")

                        if sw_params:
                            data_dict["sw_b_imf_gsm_x"] = sw_params["b_imf"][0]
                            data_dict["sw_b_imf_gsm_y"] = sw_params["b_imf"][1]
                            data_dict["sw_b_imf_gsm_z"] = sw_params["b_imf"][2]
                            data_dict["sw_v_imf_gse_x"] = sw_params["v_imf"][0]
                            data_dict["sw_v_imf_gse_y"] = sw_params["v_imf"][1]
                            data_dict["sw_v_imf_gse_z"] = sw_params["v_imf"][2]
                            data_dict["sw_np"] = sw_params["np"]
                            data_dict["sw_tp"] = sw_params["t_p"]
                            data_dict["sw_sym_h"] = sw_params["sym_h"]
                            data_dict["sw_clock_angle"] = sw_params["imf_clock_angle"]
                            data_dict["sw_p_dyn"] = sw_params["p_dyn"]

                            import math

                            bx, by, bz = (
                                sw_params["b_imf"][0],
                                sw_params["b_imf"][1],
                                sw_params["b_imf"][2],
                            )
                            b_mag = math.sqrt(bx**2 + by**2 + bz**2)
                            if b_mag > 0:
                                data_dict["sw_cone_angle"] = (
                                    math.acos(bx / b_mag) * 180 / math.pi
                                )

                        params = _get_current_run_params()
                        was_added, existing = mjl.add_jet(
                            st.session_state["master_jets"],
                            data_dict,
                            c_time,
                            params,
                            window_minutes=2,
                        )
                        if was_added:
                            mjl.save_master_list(st.session_state["master_jets"])
                            st.toast("✅ Jet added to master list!", icon="📋")
                        else:
                            existing_time = existing.get("jet_time", "unknown")
                            st.toast(
                                f"Jet already in master list (near {existing_time})",
                                icon="ℹ️",
                            )

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
                tuple: (bool success, dict dist_rc_dict, dict sw_params)
            """
            if len(recon_models) == 0:
                st.error("Please select at least one reconnection model.")
                return False, None, None
            with st.spinner(
                f"Calculating Reconnection Models for MMS{mms_probe} at {c_time.strftime('%H:%M:%S')} (this may take a while)..."
            ):
                try:
                    model_mapping = {
                        "shear": {"var_idx": 3, "label": "Shear"},
                        "bisection": {"var_idx": 6, "label": "Bisection Field"},
                        "reconnection energy": {
                            "var_idx": 4,
                            "label": "Reconnection Energy",
                        },
                        "exhaust velocity": {"var_idx": 5, "label": "Exhaust Velocity"},
                    }

                    exact_jet_time = (
                        det.get("jet_time", c_time) if det is not None else c_time
                    )
                    trange_str = exact_jet_time.strftime("%Y-%m-%d %H:%M:%S")

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
                        return False, None, None

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
                        "cmap_list": ["viridis", "cividis", "plasma", "magma"][
                            : len(images)
                        ],
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
                        "b_grids": (res[0], res[1], res[2], res[12], res[13], res[14]),
                        "save_rc_file": save_data,
                        "rc_file_name": csv_name,
                        "rc_folder": "./",
                        "df_jet_reversal": det,
                    }

                    rx_fig, dist_rc_dict = ridge_finder_multiple_interactive(
                        **figure_inputs
                    )

                    # Record history
                    run_record = {
                        "type": "recon_models",
                        "timestamp": datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
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
                    if "sel_rxn" in st.session_state:
                        del st.session_state["sel_rxn"]
                    save_auto_session()

                    return True, dist_rc_dict, sw_params
                except Exception as e:
                    st.error(f"Error running Reconnection Models: {e}")
            return False, None, None

    # =========================================================================
    # SIDEBAR ACTION BUTTONS
    # =========================================================================

    with sidebar_actions:
        st.header("Actions")

        # --- Handle Pending Quick Run ---
        if "pending_quick_run" in st.session_state:
            entry = st.session_state.pop("pending_quick_run")
            from dateutil import parser
            import pytz

            c_time = parser.parse(st.session_state["crossing_time_str"])
            if c_time.tzinfo is None:
                c_time = c_time.replace(tzinfo=pytz.utc)

            s1, det = run_jet_check(c_time, skip_master_add=True)
            if s1:
                s2, *_ = run_recon_models(c_time, det=det)
                if s2:
                    st.success(
                        f"Models successfully generated for jet at {entry.get('jet_time', 'N/A')}!"
                    )
                else:
                    st.error(
                        f"Failed to run models for jet at {entry.get('jet_time', 'N/A')}."
                    )
            else:
                st.error("Jet check failed, cannot run models.")

        # --- Feature #5: Duplicate Dialog ---
        # Check for nearby jets before running, show dialog if found
        if st.button("Run Jet Reversal Check", width="stretch"):
            if crossing_time:
                # Check for nearby jet in master list (Feature #5)
                nearby = mjl.find_nearby_jet(
                    st.session_state["master_jets"], crossing_time, window_minutes=2
                )
                if nearby is not None:
                    st.session_state["duplicate_dialog_state"] = {
                        "nearby_jet": nearby,
                        "crossing_time": crossing_time,
                        "action": "jet_check",
                    }
                    st.rerun()
                else:
                    success, jet_det = run_jet_check(crossing_time)
                    if success:
                        if jet_det:
                            st.toast(
                                "✅ Jet Reversal check completed! Jet found.", icon="✅"
                            )
                        else:
                            st.toast(
                                "❌ Jet Reversal check completed. No jet found at this exact time.",
                                icon="❌",
                            )
            else:
                st.error("Invalid crossing time.")

        if st.button("Find Next Jet", width="stretch"):
            if crossing_time:
                current_test_time = crossing_time
                found = False
                for attempt in range(max_attempts):
                    current_test_time = current_test_time + datetime.timedelta(
                        minutes=t_delta
                    )

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
                # Check for nearby jet in master list (Feature #5)
                nearby = mjl.find_nearby_jet(
                    st.session_state["master_jets"], crossing_time, window_minutes=2
                )
                if nearby is not None:
                    st.session_state["duplicate_dialog_state"] = {
                        "nearby_jet": nearby,
                        "crossing_time": crossing_time,
                        "action": "recon_models",
                    }
                    st.rerun()
                else:
                    success, _, _ = run_recon_models(crossing_time)
                    if success:
                        st.success("Reconnection Models generated!")
            else:
                st.error("Invalid crossing time.")

        if st.button("Run All", type="primary", width="stretch"):
            if crossing_time:
                # Check for nearby jet in master list (Feature #5)
                nearby = mjl.find_nearby_jet(
                    st.session_state["master_jets"], crossing_time, window_minutes=2
                )
                if nearby is not None:
                    st.session_state["duplicate_dialog_state"] = {
                        "nearby_jet": nearby,
                        "crossing_time": crossing_time,
                        "action": "run_all",
                    }
                    st.rerun()
                else:
                    s1, det = run_jet_check(crossing_time)
                    s2, _, _ = run_recon_models(crossing_time, det=det)
                    if s1 and s2:
                        st.success("All checks and models generated successfully!")
            else:
                st.error("Invalid crossing time.")

        if st.button(
            "Run Statistics Mode", width="stretch", disabled=not use_stats_mode
        ):
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
                    t_delta_td = datetime.timedelta(minutes=stats_delta_mins)

                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    file_times = []
                    if stop_condition == "End Time":
                        total_steps = (
                            int(
                                (stats_end_time - crossing_time).total_seconds()
                                // (stats_delta_mins * 60)
                            )
                            + 1
                        )
                    elif stop_condition == "From File List":
                        if stats_uploaded_file is not None:
                            try:
                                from dateutil import parser
                                import pytz

                                content = (
                                    stats_uploaded_file.getvalue()
                                    .decode("utf-8")
                                    .splitlines()
                                )
                                for line in content:
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
                                    st.error(
                                        "No valid times found in the uploaded file."
                                    )
                                    st.stop()
                                total_steps = len(file_times)
                            except Exception as e:
                                st.error(f"Error parsing uploaded file: {e}")
                                st.stop()
                        else:
                            st.error("Please upload a file list first.")
                            st.stop()
                    else:
                        total_steps = target_jets

                    step = 0
                    jets_found = 0

                    while True:
                        if stop_condition == "From File List":
                            if step >= total_steps:
                                break
                            curr_t = file_times[step]
                        elif stop_condition == "End Time" and curr_t > stats_end_time:
                            break
                        elif (
                            stop_condition == "Target Number of Jets"
                            and jets_found >= target_jets
                        ):
                            break

                        # fail-safe for target jets to avoid infinite loop
                        if (
                            stop_condition == "Target Number of Jets"
                            and step > target_jets * 100
                        ):
                            st.warning(
                                "Reached maximum search attempts (100x target). Stopping early."
                            )
                            break

                        if stop_condition in ["End Time", "From File List"]:
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
                            latest_jet_fig = st.session_state["history"]["jet_checks"][
                                -1
                            ].get("fig")
                            if latest_jet_fig:
                                with live_jet_plot_placeholder.container():
                                    st.plotly_chart(
                                        latest_jet_fig,
                                        width="stretch",
                                        key=f"live_jet_{step}",
                                    )

                        if s1 and det:
                            run_recon_models(
                                curr_t, save_data=True, csv_name=stats_csv_name, det=det
                            )
                            jets_found += 1

                        step += 1
                        if stop_condition in ["End Time", "From File List"]:
                            progress_bar.progress(min(step / total_steps, 1.0))
                        else:
                            progress_bar.progress(min(jets_found / target_jets, 1.0))

                        if stop_condition != "From File List":
                            curr_t += t_delta_td

                    status_text.text("Statistics Mode Completed!")
                    if stop_condition in ["End Time", "From File List"]:
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

    # =========================================================================
    # DUPLICATE JET DIALOG (Feature #5)
    # =========================================================================

    # Handle duplicate dialog state — shown on the Controls tab as a prominent block
    with tab_controls:
        if st.session_state.get("duplicate_dialog_state") is not None:
            dialog_state = st.session_state["duplicate_dialog_state"]
            nearby = dialog_state["nearby_jet"]
            c_time = dialog_state["crossing_time"]
            action = dialog_state["action"]

            st.warning(
                f"⚠️ **A jet is already present at {nearby.get('jet_time', 'unknown')}** "
                f"(within 2 minutes of your input time "
                f"{c_time.strftime('%Y-%m-%d %H:%M:%S')}). "
                f"What would you like to do?"
            )

            col_a, col_b, col_c, col_d = st.columns(4)
            if col_a.button("Generate Jet Reversal Plot Only", key="dup_jet_only"):
                st.session_state["duplicate_dialog_state"] = None
                success, jet_det = run_jet_check(c_time, skip_master_add=True)
                if success:
                    if jet_det:
                        st.toast("✅ Jet Reversal plot generated!", icon="✅")
                    else:
                        st.toast(
                            "❌ Jet Reversal check completed. No jet found.", icon="❌"
                        )
                st.rerun()

            if col_b.button("Generate Reconnection Model Only", key="dup_rxn_only"):
                st.session_state["duplicate_dialog_state"] = None
                run_recon_models(c_time)
                st.toast("✅ Reconnection Model generated!", icon="✅")
                st.rerun()

            if col_c.button("Generate Both", key="dup_both"):
                st.session_state["duplicate_dialog_state"] = None
                s1, det = run_jet_check(c_time, skip_master_add=True)
                s2 = run_recon_models(c_time, det=det)
                if s1 and s2:
                    st.toast("✅ Both plots generated!", icon="✅")
                st.rerun()

            if col_d.button("Skip / Cancel", key="dup_skip"):
                st.session_state["duplicate_dialog_state"] = None
                st.rerun()

    # =========================================================================
    # JET REVERSAL TAB
    # =========================================================================

    def get_dropdown_options(history_list):
        """
        Generates the list of available dates dynamically based on the master jet list history.
        """
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
            selected_option = col_sel.selectbox(
                "Browse Run History", options, index=curr_idx
            )
            new_idx = options.index(selected_option)

            if (
                col_prev.button("Previous Run", key="prev_jet", width="stretch")
                and curr_idx > 0
            ):
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

    # =========================================================================
    # RECONNECTION MODELS TAB
    # =========================================================================

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

            if (
                col_prev.button("Previous Run", key="prev_rxn", width="stretch")
                and curr_idx > 0
            ):
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

    # =========================================================================
    # MASTER JET LIST TAB (Features #7, #8, #9, #13)
    # =========================================================================

    with tab_master:
        st.header("Master Jet List")
        st.markdown(
            "This is the persistent record of all observed reconnection jets. "
            "Jets within 2 minutes of each other are treated as duplicates."
        )

        master_jets = st.session_state["master_jets"]

        if len(master_jets) > 0:
            # Build display DataFrame (Feature #7)
            display_data = []
            for i, entry in enumerate(master_jets):
                # Parse date and times
                j_time_str = entry.get("jet_time", "N/A")
                c_time_str = entry.get("crossing_time", "N/A")
                date_val = "N/A"
                j_time_val = j_time_str
                c_time_val = c_time_str

                try:
                    if j_time_str != "N/A":
                        dt_j = datetime.datetime.fromisoformat(
                            j_time_str.replace(" ", "T")
                        )
                        date_val = dt_j.strftime("%Y-%m-%d")
                        j_time_val = dt_j.strftime("%H:%M:%S")
                    if c_time_str != "N/A":
                        dt_c = datetime.datetime.fromisoformat(
                            c_time_str.replace(" ", "T")
                        )
                        c_time_val = dt_c.strftime("%H:%M:%S")
                except:
                    pass

                # Position
                x = entry.get("data_x_gsm")
                y = entry.get("data_y_gsm")
                z = entry.get("data_z_gsm")
                if x is not None and y is not None and z is not None:
                    try:
                        import math

                        x, y, z = float(x), float(y), float(z)
                        r = math.sqrt(x**2 + y**2 + z**2)
                        pos_str = f"[{x:.2f}, {y:.2f}, {z:.2f}, {r:.2f}]"
                    except:
                        pos_str = "N/A"
                else:
                    pos_str = "N/A"

                # Solar Wind B
                bx = entry.get("data_sw_b_imf_gsm_x")
                by = entry.get("data_sw_b_imf_gsm_y")
                bz = entry.get("data_sw_b_imf_gsm_z")
                if bx is not None and by is not None and bz is not None:
                    try:
                        bx, by, bz = float(bx), float(by), float(bz)
                        b_mag = math.sqrt(bx**2 + by**2 + bz**2)
                        b_str = f"[{bx:.1f}, {by:.1f}, {bz:.1f}, {b_mag:.1f}]"
                    except:
                        b_str = "N/A"
                else:
                    b_str = "N/A"

                # Solar Wind V
                vx = entry.get("data_sw_v_imf_gse_x")
                vy = entry.get("data_sw_v_imf_gse_y")
                vz = entry.get("data_sw_v_imf_gse_z")
                if vx is not None and vy is not None and vz is not None:
                    try:
                        vx, vy, vz = float(vx), float(vy), float(vz)
                        v_mag = math.sqrt(vx**2 + vy**2 + vz**2)
                        v_str = f"[{vx:.1f}, {vy:.1f}, {vz:.1f}, {v_mag:.1f}]"
                    except:
                        v_str = "N/A"
                else:
                    v_str = "N/A"

                # SW Np & Tp
                np_val = entry.get("data_sw_np", "N/A")
                if isinstance(np_val, (float, int)):
                    np_val = f"{np_val:.2f}"
                tp_val = entry.get("data_sw_tp", "N/A")
                if isinstance(tp_val, (float, int)):
                    tp_val = f"{tp_val:.1f}"

                # Clock Angle, P_dyn, Cone Angle
                clock_angle = entry.get("data_sw_clock_angle", "N/A")
                if isinstance(clock_angle, (float, int)):
                    clock_angle = f"{clock_angle:.1f}"
                p_dyn = entry.get("data_sw_p_dyn", "N/A")
                if isinstance(p_dyn, (float, int)):
                    p_dyn = f"{p_dyn:.2f}"
                cone_angle = entry.get(
                    "data_sw_cone_angle", entry.get("data_cone_angle", "N/A")
                )
                if isinstance(cone_angle, (float, int)):
                    cone_angle = f"{cone_angle:.1f}"

                def _fmt_rc(key_name):
                    """
                    Helper function to format model names for table headers.
                    """
                    val = entry.get(key_name)
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        return "N/A"
                    return f"{val:.2f}"

                # Compute average Recon. Dist. if models exist
                rc_keys = [
                    "data_r_rc_Shear",
                    "data_r_rc_Bisection Field",
                    "data_r_rc_Exhaust Velocity",
                    "data_r_rc_Reconnection Energy",
                ]
                rc_vals = []
                for k in rc_keys:
                    v = entry.get(k)
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        rc_vals.append(v)

                recon_dist_str = (
                    f"{sum(rc_vals)/len(rc_vals):.2f}" if rc_vals else "N/A"
                )

                # Format Model Parameters
                mod_params = (
                    f"Model: {entry.get('tsy_model', 'N/A')}, "
                    f"m_p: {entry.get('m_p', 'N/A')}, "
                    f"dr: {entry.get('dr', 'N/A')}"
                )

                def fmt_val(v, dec=2):
                    """
                    Formats a float value to two decimal places safely, handling NaNs.
                    """
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        return "N/A"
                    return f"{v:.{dec}f}"

                def fmt_vec(vx, vy, vz, mag=None):
                    """
                    Formats a 3D vector array into a clean string representation.
                    """
                    if any(
                        v is None or (isinstance(v, float) and math.isnan(v))
                        for v in [vx, vy, vz]
                    ):
                        return "N/A"
                    if mag is None or (isinstance(mag, float) and math.isnan(mag)):
                        mag = math.sqrt(vx**2 + vy**2 + vz**2)
                    return f"({vx:.2f}, {vy:.2f}, {vz:.2f}, {mag:.2f})"

                mms_pos = fmt_vec(
                    entry.get("data_x_gsm"),
                    entry.get("data_y_gsm"),
                    entry.get("data_z_gsm"),
                    entry.get("data_r_spc"),
                )
                imf_b = fmt_vec(
                    entry.get("data_sw_b_imf_gsm_x"),
                    entry.get("data_sw_b_imf_gsm_y"),
                    entry.get("data_sw_b_imf_gsm_z"),
                )
                sw_vel = fmt_vec(
                    entry.get("data_sw_v_imf_gse_x"),
                    entry.get("data_sw_v_imf_gse_y"),
                    entry.get("data_sw_v_imf_gse_z"),
                )

                jet_time_val = entry.get("data_jet_time", c_time_val)
                if isinstance(jet_time_val, (pd.Timestamp, datetime.datetime)):
                    jet_time_val = jet_time_val.strftime("%H:%M:%S")
                elif isinstance(jet_time_val, str):
                    jet_time_val = (
                        jet_time_val.split(" ")[1]
                        if " " in jet_time_val
                        else jet_time_val
                    )
                    jet_time_val = jet_time_val.split("+")[0]

                if " " in c_time_str:
                    date_val = c_time_str.split(" ")[0]
                elif "T" in c_time_str:
                    date_val = c_time_str.split("T")[0]
                else:
                    # Fallback to the previously parsed date_val
                    pass

                row = {
                    "Select": False,
                    "Event Index": i + 1,
                    "Date": date_val,
                    "Time": jet_time_val,
                    "MMS Probe": entry.get("mms_probe", entry.get("data_Probe", "N/A")),
                    "MMS Pos [R_E] (GSM)": mms_pos,
                    "IMF B [nT] (GSM)": imf_b,
                    "SW Dyn. Pressure [nPa]": fmt_val(entry.get("data_sw_p_dyn")),
                    "Sym-H [nT]": fmt_val(entry.get("data_sw_sym_h")),
                    "Plasma Vel [km/s] (GSE)": sw_vel,
                    "Clock Angle [deg]": clock_angle,
                    "Cone Angle [deg]": cone_angle,
                    "Shear [Re]": fmt_val(
                        entry.get("data_r_rc_Shear")
                    ),
                    "Recon. Eng [Re]": fmt_val(
                        entry.get("data_r_rc_Reconnection Energy")
                    ),
                    "Exhaust Vel. [Re]": fmt_val(
                        entry.get("data_r_rc_Exhaust Velocity")
                    ),
                    "Bisec Field [Re]": fmt_val(
                        entry.get("data_r_rc_Bisection Field")
                    ),
                    "Model Parameters": mod_params,
                    "_original_index": i,
                }
                display_data.append(row)

            display_df = pd.DataFrame(display_data)

            # Show total count
            st.metric("Total Jets in Master List", len(master_jets))

            # --- Sorting Logic ---
            sort_options = [
                c
                for c in display_df.columns
                if c not in ["Select", "#", "_original_index"]
            ]
            sort_by = st.selectbox(
                "Sort Master List By:", options=["None"] + sort_options
            )

            if sort_by != "None":
                # Convert "N/A" to real NaNs for proper sorting, then sort, then convert back
                temp_df = display_df.replace("N/A", float("nan"))
                temp_df = temp_df.sort_values(
                    by=sort_by, ascending=True, na_position="last"
                ).reset_index(drop=True)
                display_df = temp_df.replace(float("nan"), "N/A")

                # Highlight the sorted column by renaming it
                highlight_name = f"{sort_by} 🔽"
                display_df = display_df.rename(columns={sort_by: highlight_name})
            else:
                highlight_name = None

            # Feature #8: Editable dataframe with selection for pruning
            # Build column config dynamically
            col_config = {
                "Select": st.column_config.CheckboxColumn(
                    "Select", help="Select rows to delete", default=False
                ),
                "_original_index": None,  # Hide this column
            }

            if show_hints:
                col_helps = {
                    "MMS Pos [R_E]": "MMS Position Vector (X, Y, Z, R) in R_E GSM coordinates",
                    "IMF B [nT]": "Interplanetary Magnetic Field Vector (X, Y, Z, |B|) in nT",
                    "Plasma Vel [km/s]": "Solar Wind Plasma Velocity Vector (X, Y, Z, |V|) in km/s",
                    "Event Index": "Index of the event in the master list",
                    "Date": "Date of the crossing",
                    "Time": "Time of the jet",
                    "MMS Probe": "Which MMS spacecraft recorded the data",
                    "SW Dyn. Pressure [nPa]": "Solar Wind Dynamic Pressure",
                    "Sym-H [nT]": "Sym-H index",
                    "Clock Angle [deg]": "IMF Clock Angle",
                    "Cone Angle [deg]": "IMF Cone Angle",
                    "Shear [Re]": "Distance to the Maximum Magnetic Shear X-line",
                    "Recon. Eng [Re]": "Distance to the Maximum Reconnection Energy X-line",
                    "Exhaust Vel. [Re]": "Distance to the Maximum Exhaust Velocity X-line",
                    "Bisec Field [Re]": "Distance to the Minimum Bisection Field X-line",
                    "Model Parameters": "Tsyganenko model and grid parameters used for these results",
                }
            else:
                col_helps = {}

            # Make all other columns disabled
            for col in display_df.columns:
                if col not in ["Select", "_original_index"]:
                    base_col = col.replace(" 🔽", "")
                    help_str = col_helps.get(base_col, None)
                    col_config[col] = st.column_config.TextColumn(
                        col, disabled=True, help=help_str
                    )

            edited_df = st.data_editor(
                display_df,
                hide_index=True,
                width="stretch",
                column_config=col_config,
                key="master_list_editor",
            )

            # --- Manual Pruning (Feature #8) ---
            selected_rows = edited_df[edited_df["Select"]]
            if not selected_rows.empty:
                selected_original_indices = selected_rows["_original_index"].tolist()
                st.warning(
                    f"{len(selected_original_indices)} jet(s) selected for deletion."
                )
                if st.button(
                    f"🗑️ Delete {len(selected_original_indices)} Selected Jet(s)",
                    type="primary",
                    key="delete_selected_jets",
                ):
                    mjl.delete_jets(
                        st.session_state["master_jets"], selected_original_indices
                    )
                    mjl.save_master_list(st.session_state["master_jets"])
                    st.success(
                        f"Deleted {len(selected_original_indices)} jet(s) from master list."
                    )
                    st.rerun()

            # --- Export Options (Feature #9) ---
            st.markdown("---")
            st.subheader("Export Master List")
            col_csv, col_json, col_pkl = st.columns(3)

            csv_bytes = mjl.export_to_csv(master_jets)
            col_csv.download_button(
                label="📄 Export as CSV",
                data=csv_bytes,
                file_name=f"master_jets_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width="stretch",
            )

            json_bytes = mjl.export_to_json(master_jets)
            col_json.download_button(
                label="📋 Export as JSON",
                data=json_bytes,
                file_name=f"master_jets_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                width="stretch",
            )

            pkl_bytes = mjl.export_to_pickle(master_jets)
            col_pkl.download_button(
                label="📦 Export as Pickle",
                data=pkl_bytes,
                file_name=f"master_jets_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                mime="application/octet-stream",
                width="stretch",
            )

            # --- Quick Re-run (Feature #13) ---
            st.markdown("---")
            st.subheader("Quick Re-run")
            st.markdown(
                "Select a jet from the list below to load its parameters into the dashboard."
            )

            rerun_options = [
                f"Jet {i+1}: {entry.get('jet_time', 'N/A')} (Probe {entry.get('mms_probe', '?')})"
                for i, entry in enumerate(master_jets)
            ]
            selected_rerun = st.selectbox(
                "Select jet to load",
                ["(none)"] + rerun_options,
                key="quick_rerun_selector",
            )

            col1, col2 = st.columns(2)
            load_only = col1.button(
                "🔄 Load into Dashboard",
                key="load_jet_to_dashboard",
                use_container_width=True,
            )
            load_and_run = col2.button(
                "🚀 Load & Run Models", key="load_and_run_jet", use_container_width=True
            )

            if load_only or load_and_run:
                if selected_rerun and selected_rerun != "(none)":
                    rerun_idx = rerun_options.index(selected_rerun)
                    entry = master_jets[rerun_idx]

                    # Populate session state with the jet's parameters
                    ct = entry.get("crossing_time", None)
                    if ct:
                        st.session_state["crossing_time_str"] = str(ct).replace(
                            "+00:00", ""
                        )
                        # Handle ISO format with T separator
                        if "T" in st.session_state["crossing_time_str"]:
                            st.session_state["crossing_time_str"] = st.session_state[
                                "crossing_time_str"
                            ].replace("T", " ")
                        # Truncate to seconds precision
                        st.session_state["crossing_time_str"] = st.session_state[
                            "crossing_time_str"
                        ][:19]

                    # Load sidebar parameters from the entry
                    param_map = {
                        "mms_probe": "preset_mms_probe",
                        "dt": "preset_dt",
                        "jet_len": "preset_jet_len",
                        "data_rate": "preset_data_rate",
                        "level": "preset_level",
                        "coord_type": "preset_coord_type",
                        "time_clip": "preset_time_clip",
                        "tsy_model": "preset_tsy_model",
                        "recon_models": "preset_recon_models",
                        "omni_level": "preset_omni_level",
                        "m_p": "preset_m_p",
                        "dr": "preset_dr",
                        "limits": "preset_limits",
                    }
                    for src_key, dst_key in param_map.items():
                        if src_key in entry:
                            st.session_state[dst_key] = entry[src_key]

                    save_auto_session()

                    if load_and_run:
                        st.session_state["pending_quick_run"] = entry
                        st.rerun()
                    else:
                        st.session_state["show_toast"] = (
                            f"Loaded parameters from jet at {entry.get('jet_time', 'N/A')}."
                        )
                        st.rerun()

            # --- Import Master List ---
            st.markdown("---")
            st.subheader("Import Master List")
            uploaded_master = st.file_uploader(
                "Upload a master list file",
                type=["json", "csv", "pkl"],
                key="import_master_list",
            )
            if uploaded_master is not None:
                if st.button("Import and Merge", key="import_merge_master"):
                    try:
                        if uploaded_master.name.endswith(".json"):
                            imported = json.load(uploaded_master)
                        elif uploaded_master.name.endswith(".pkl"):
                            imported = pickle.load(uploaded_master)
                        elif uploaded_master.name.endswith(".csv"):
                            df_imported = pd.read_csv(uploaded_master)
                            imported = df_imported.to_dict(orient="records")
                        else:
                            imported = []

                        added_count = 0
                        for entry in imported:
                            jet_time = entry.get("jet_time", entry.get("crossing_time"))
                            if jet_time:
                                existing = mjl.find_nearby_jet(
                                    st.session_state["master_jets"],
                                    jet_time,
                                    window_minutes=2,
                                )
                                if existing is None:
                                    st.session_state["master_jets"].append(entry)
                                    added_count += 1

                        mjl.save_master_list(st.session_state["master_jets"])
                        st.success(
                            f"Imported {added_count} new jet(s) (skipped duplicates)."
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error importing master list: {e}")

        else:
            st.info(
                "No jets in the master list yet. Run a Jet Reversal Check to add detected jets."
            )

    # =========================================================================
    # STATISTICS TAB (Features #6, #11)
    # =========================================================================

    if tab_stats is not None:
        with tab_stats:
            st.header("Statistics Results")

            st.markdown(
                "Generate statistical plots from batch runs or uploaded CSV data."
            )

            # CSV handling
            default_csv = st.session_state.get("latest_stats_csv", "")
            csv_source = st.radio(
                "CSV Source",
                [
                    "Use generated from Statistics Mode",
                    "Upload existing CSV",
                    "Generate from Master Jet List",
                ],
            )

            csv_path = None
            stats_df = None  # For dynamic filtering

            if csv_source == "Use generated from Statistics Mode":
                if os.path.exists(default_csv):
                    csv_path = default_csv
                    st.success(f"Using `{default_csv}`")
                    with open(csv_path, "rb") as f:
                        st.download_button(
                            "Download CSV",
                            data=f,
                            file_name=default_csv,
                            mime="text/csv",
                        )
                    stats_df = pd.read_csv(csv_path, index_col=False)
                else:
                    st.warning("No generated CSV found. Run Statistics Mode first.")

            elif csv_source == "Upload existing CSV":
                uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
                if uploaded_csv is not None:
                    csv_path = "temp_uploaded_stats.csv"
                    with open(csv_path, "wb") as f:
                        f.write(uploaded_csv.getbuffer())
                    st.success("CSV Uploaded successfully.")
                    stats_df = pd.read_csv(csv_path, index_col=False)

            elif csv_source == "Generate from Master Jet List":
                # Feature #6: Generate stats from master list
                if len(st.session_state["master_jets"]) > 0:
                    st.info(
                        f"Master list has {len(st.session_state['master_jets'])} jet(s). "
                        f"Use the filters below or generate directly."
                    )
                    ml_col1, ml_col2 = st.columns(2)
                    ml_start = ml_col1.text_input(
                        "Filter Start Time (optional)",
                        value="",
                        help="Leave blank to include all jets.",
                        key="ml_stats_start",
                    )
                    ml_end = ml_col2.text_input(
                        "Filter End Time (optional)",
                        value="",
                        help="Leave blank to include all jets.",
                        key="ml_stats_end",
                    )

                    ml_start_dt = None
                    ml_end_dt = None
                    try:
                        if ml_start.strip():
                            ml_start_dt = datetime.datetime.strptime(
                                ml_start.strip(), "%Y-%m-%d %H:%M:%S"
                            ).replace(tzinfo=pytz.utc)
                    except ValueError:
                        st.warning("Invalid start time format.")
                    try:
                        if ml_end.strip():
                            ml_end_dt = datetime.datetime.strptime(
                                ml_end.strip(), "%Y-%m-%d %H:%M:%S"
                            ).replace(tzinfo=pytz.utc)
                    except ValueError:
                        st.warning("Invalid end time format.")

                    result_df = mjl.master_list_to_stats_csv(
                        st.session_state["master_jets"],
                        time_start=ml_start_dt,
                        time_end=ml_end_dt,
                    )
                    if result_df is not None and len(result_df) > 0:
                        # Save to a temp CSV for the plotting functions
                        temp_ml_csv = "temp_master_list_stats.csv"
                        result_df.to_csv(temp_ml_csv, index=False)
                        csv_path = temp_ml_csv
                        stats_df = pd.read_csv(csv_path, index_col=False)
                        st.success(
                            f"Generated stats CSV from {len(result_df)} master list entries."
                        )
                    else:
                        st.warning(
                            "No matching entries found in master list for the given range."
                        )
                else:
                    st.warning(
                        "Master Jet List is empty. Run jet checks to populate it first."
                    )

            # --- Dynamic Filtering (Feature #11) ---
            if stats_df is not None and csv_path and os.path.exists(csv_path):
                st.markdown("---")
                st.subheader("Dynamic Filters")
                st.caption(
                    "Filter the data before generating plots. Only rows matching ALL "
                    "filters will be included."
                )

                filter_col1, filter_col2, filter_col3 = st.columns(3)

                # BZ filter
                if "b_imf_z" in stats_df.columns:
                    bz_min_val = float(stats_df["b_imf_z"].min())
                    bz_max_val = float(stats_df["b_imf_z"].max())
                    if bz_min_val < bz_max_val:
                        bz_range = filter_col1.slider(
                            "IMF Bz Range [nT]",
                            min_value=bz_min_val,
                            max_value=bz_max_val,
                            value=(bz_min_val, bz_max_val),
                            key="filter_bz",
                        )
                    else:
                        bz_range = (bz_min_val, bz_max_val)
                else:
                    bz_range = None

                # Dynamic pressure filter
                if "p_dyn" in stats_df.columns:
                    pdyn_min_val = float(stats_df["p_dyn"].min())
                    pdyn_max_val = float(stats_df["p_dyn"].max())
                    if pdyn_min_val < pdyn_max_val:
                        pdyn_range = filter_col2.slider(
                            "Dynamic Pressure Range [nPa]",
                            min_value=pdyn_min_val,
                            max_value=pdyn_max_val,
                            value=(pdyn_min_val, pdyn_max_val),
                            key="filter_pdyn",
                        )
                    else:
                        pdyn_range = (pdyn_min_val, pdyn_max_val)
                else:
                    pdyn_range = None

                # Shear angle filter
                shear_col = None
                for candidate in ["msh_msp_shear", "angle_b_lmn_vec_msp_msh_median"]:
                    if candidate in stats_df.columns:
                        shear_col = candidate
                        break

                if shear_col:
                    shear_min_val = float(stats_df[shear_col].min())
                    shear_max_val = float(stats_df[shear_col].max())
                    if shear_min_val < shear_max_val:
                        shear_range = filter_col3.slider(
                            "Shear Angle Range [°]",
                            min_value=shear_min_val,
                            max_value=shear_max_val,
                            value=(shear_min_val, shear_max_val),
                            key="filter_shear",
                        )
                    else:
                        shear_range = (shear_min_val, shear_max_val)
                else:
                    shear_range = None

                # Apply filters
                filtered_df = stats_df.copy()
                if bz_range is not None and "b_imf_z" in filtered_df.columns:
                    filtered_df = filtered_df[
                        (filtered_df["b_imf_z"] >= bz_range[0])
                        & (filtered_df["b_imf_z"] <= bz_range[1])
                    ]
                if pdyn_range is not None and "p_dyn" in filtered_df.columns:
                    filtered_df = filtered_df[
                        (filtered_df["p_dyn"] >= pdyn_range[0])
                        & (filtered_df["p_dyn"] <= pdyn_range[1])
                    ]
                if shear_range is not None and shear_col in filtered_df.columns:
                    filtered_df = filtered_df[
                        (filtered_df[shear_col] >= shear_range[0])
                        & (filtered_df[shear_col] <= shear_range[1])
                    ]

                st.caption(
                    f"Showing **{len(filtered_df)}** of **{len(stats_df)}** rows after filtering."
                )

                # Write filtered data to a temp CSV for the plotting functions
                filtered_csv_path = "temp_filtered_stats.csv"
                filtered_df.to_csv(filtered_csv_path, index=False)

                st.success(f"Using statistics data from: {csv_path}")

                # Plot selection
                # Plot selection
                base_var_options = {
                    "b_imf_z": r"IMF $B_z$ [nT]",
                    "b_imf_x": r"IMF $B_x$ [nT]",
                    "b_imf_y": r"IMF $B_y$ [nT]",
                    "B_imf_gsm_z": r"IMF $B_z$ [nT]",
                    "B_imf_gsm_x": r"IMF $B_x$ [nT]",
                    "B_imf_gsm_y": r"IMF $B_y$ [nT]",
                    "sw_b_imf_gsm_z": r"IMF $B_z$ [nT]",
                    "sw_b_imf_gsm_x": r"IMF $B_x$ [nT]",
                    "sw_b_imf_gsm_y": r"IMF $B_y$ [nT]",
                    "imf_clock_angle": r"IMF Clock Angle ($^\circ$)",
                    "cone_angle": r"Cone Angle ($\cos^{-1}(B_x/|\mathbf{B}|)$) [$^\circ$]",
                    "p_dyn": r"Dynamic Pressure [nPa]",
                    "msh_msp_shear": r"Shear Angle ($^\circ$)",
                    "r_rc": r"Reconnection Distance [$R_E$]",
                    "delta_beta": r"$\Delta \beta$",
                    "bb": r"IMF $B_y/|\mathbf{B}|$",
                    "beta_msh_mean": r"$\beta_{\rm p}$",
                    "np_msp_median": r"$N_p$ (MSP) [cm$^{-3}$]",
                    "tp_para_msp_median": r"$Tp_{\parallel}$ [$10^6$ K]",
                    "tp_perp_msp_median": r"$Tp_{\perp}$ [$10^6$ K]",
                }

                # Dynamically load from key_list.md
                key_list_path = os.path.join(os.path.dirname(__file__), "key_list.md")
                if os.path.exists(key_list_path):
                    with open(key_list_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line or ":" not in line:
                                continue
                            keys_part, labels_part = line.split(":", 1)
                            keys = [k.strip() for k in keys_part.split(",")]
                            labels = [l.strip() for l in labels_part.split(",")]

                            full_keys = []
                            base_key = ""
                            for k in keys:
                                if k.startswith("_"):
                                    full_keys.append(base_key + k)
                                else:
                                    full_keys.append(k)
                                    if k.endswith("_l"):
                                        base_key = k[:-2]
                                    else:
                                        base_key = ""

                            for k, l in zip(full_keys, labels):
                                if k.startswith("data_"):
                                    k = k[5:]
                                base_var_options[k] = l

                var_options = {}
                import numpy as np

                numeric_cols = stats_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col == "method_used" or col.startswith("Unnamed"):
                        continue
                    if stats_df[col].isna().all():
                        continue
                    var_options[col] = base_var_options.get(
                        col, col.replace("_", " ").title()
                    )

                for model_col in ["r_rc_Shear", "r_rc_Bisection Field", "r_rc_Reconnection Energy", "r_rc_Exhaust Velocity"]:
                    if model_col in stats_df.columns or f"data_{model_col}" in stats_df.columns:
                        var_options["r_rc"] = base_var_options.get(
                            "r_rc", "Reconnection Distance [$R_E$]"
                        )
                        break

                available_vars = list(var_options.keys())
                default_x = (
                    ["b_imf_z"]
                    if "b_imf_z" in available_vars
                    else ([available_vars[0]] if available_vars else [])
                )
                default_y = (
                    ["b_imf_y"]
                    if "b_imf_y" in available_vars
                    else ([available_vars[1]] if len(available_vars) > 1 else default_x)
                )

                plots_to_gen = st.multiselect(
                    "Select Figures to Generate",
                    [
                        "Histograms",
                        "KDE Plots",
                        "2D Histograms",
                        "Scatter Plots",
                        "MMS Location Scatter Plot",
                        "Seaborn Joint-Plots",
                    ],
                    default=["Seaborn Joint-Plots"],
                )

                # Extra option for seaborn plots
                marker_size_var = "None"
                if "Seaborn Joint-Plots" in plots_to_gen:
                    marker_options = ["None"] + available_vars
                    default_marker = "r_rc" if "r_rc" in available_vars else "None"
                    marker_size_var = st.selectbox(
                        "Variable for Marker Size (Z-parameter)",
                        marker_options,
                        format_func=lambda x: (
                            var_options.get(x, x)
                            if x != "None"
                            else "Constant (No Scaling)"
                        ),
                        index=marker_options.index(default_marker),
                    )

                x_vars = st.multiselect(
                    "X-Axis Variable(s)",
                    available_vars,
                    format_func=lambda x: var_options[x],
                    default=default_x,
                    key="stats_x_vars_v2",
                )
                y_vars = st.multiselect(
                    "Y-Axis Variable(s)",
                    available_vars,
                    format_func=lambda x: var_options[x],
                    default=default_y,
                    key="stats_y_vars_v2",
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
                            # --- Interactive Plotly figures ---
                            plotly_plots = [
                                p for p in plots_to_gen if p != "Seaborn Joint-Plots"
                            ]
                            if plotly_plots:
                                for x_var, y_var in zip(x_vars, y_vars):
                                    if len(x_vars) > 1:
                                        st.markdown(
                                            f"### {var_options[x_var]} vs {var_options[y_var]}"
                                        )

                                    figures, err = generate_interactive_plots(
                                        filtered_csv_path,
                                        dark_mode=is_dark_mode,
                                        selected_plots=plotly_plots,
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
                                                st.plotly_chart(
                                                    fig,
                                                    width="stretch",
                                                )
                                            else:
                                                st.warning(
                                                    f"Could not generate {title}"
                                                )

                            # --- Static Seaborn joint-plots ---
                            if "Seaborn Joint-Plots" in plots_to_gen:
                                for x_var, y_var in zip(x_vars, y_vars):
                                    st.markdown(
                                        f"### Seaborn Joint-Plot: "
                                        f"{var_options.get(x_var, x_var)} vs "
                                        f"{var_options.get(y_var, y_var)}"
                                    )
                                    try:
                                        sb_fig = generate_seaborn_jointplots(
                                            df_full=filtered_df,
                                            x_key=x_var,
                                            y_key=y_var,
                                            x_label=var_options.get(x_var, x_var),
                                            y_label=var_options.get(y_var, y_var),
                                            dark_mode=is_dark_mode,
                                            marker_size_var=marker_size_var,
                                        )
                                        st.pyplot(sb_fig)
                                    except Exception as e:
                                        st.error(
                                            f"Error generating Seaborn Joint-Plot: {e}"
                                        )


if __name__ == "__main__":
    main()
