"""
Master Jet List Management Module

Provides persistent storage and management of observed reconnection jet events.
The master list is stored as a JSON file at ~/.rxn_location_master_jets.json,
enabling deduplication (2-minute window), export, and quick re-run functionality.
"""

import json
import os
import datetime
import pickle
import re
from pathlib import Path

import pandas as pd
import pytz
import numpy as np

DEFAULT_MASTER_LIST_PATH = Path(os.path.expanduser("~")) / ".rxn_location_master_jets.json"


def load_master_list(path=None):
    """
    Load the master jet list from disk.

    Parameters
    ----------
    path : str or Path, optional
        Path to the JSON file. Defaults to ~/.rxn_location_master_jets.json.

    Returns
    -------
    list of dict
        Each dict represents one observed jet event with its parameters.
    """
    if path is None:
        path = DEFAULT_MASTER_LIST_PATH
    path = Path(path)
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_master_list(entries, path=None):
    """
    Save the master jet list to disk.

    Parameters
    ----------
    entries : list of dict
        The master list entries to persist.
    path : str or Path, optional
        Path to the JSON file. Defaults to ~/.rxn_location_master_jets.json.
    """
    if path is None:
        path = DEFAULT_MASTER_LIST_PATH
    path = Path(path)
    try:
        with open(path, "w") as f:
            json.dump(entries, f, indent=2, default=str)
    except IOError:
        pass


def _parse_time(time_str):
    """
    Parse a time string into a timezone-aware datetime object.

    Parameters
    ----------
    time_str : str or datetime.datetime
        The time to parse. Accepts ISO format, or 'YYYY-MM-DD HH:MM:SS' with or
        without timezone info.

    Returns
    -------
    datetime.datetime
        A UTC-aware datetime object.
    """
    if isinstance(time_str, datetime.datetime):
        if time_str.tzinfo is None:
            return time_str.replace(tzinfo=pytz.utc)
        return time_str

    # Convert to string in case of pandas Timestamp or similar
    time_str = str(time_str)

    # Truncate nanosecond-precision fractional seconds to microseconds (6 digits).
    # Python's strptime %f only supports up to 6 decimal places, but pyspedas
    # can produce 9+ digit fractional seconds (e.g. "16:47:33.709315896+00:00").
    time_str = re.sub(
        r"(\d{2}:\d{2}:\d{2})\.(\d{6})\d+",
        r"\1.\2",
        time_str,
    )

    # Try parsing common formats
    for fmt in [
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ]:
        try:
            dt = datetime.datetime.strptime(time_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=pytz.utc)
            return dt
        except ValueError:
            continue

    raise ValueError(f"Cannot parse time string: {time_str}")


def find_nearby_jet(entries, time, window_minutes=2):
    """
    Find a jet in the master list within a time window of the given time.

    Parameters
    ----------
    entries : list of dict
        The master list entries.
    time : datetime.datetime or str
        The time to search around.
    window_minutes : int or float
        The half-window size in minutes (default 2).

    Returns
    -------
    dict or None
        The matching entry if found within the window, otherwise None.
    """
    target = _parse_time(time)
    window = datetime.timedelta(minutes=window_minutes)

    closest_entry = None
    closest_delta = None

    for entry in entries:
        try:
            entry_time = _parse_time(entry["jet_time"])
            delta = abs(entry_time - target)
            if delta <= window:
                if closest_delta is None or delta < closest_delta:
                    closest_entry = entry
                    closest_delta = delta
        except (KeyError, ValueError):
            continue

    return closest_entry


def add_jet(entries, data_dict, crossing_time, params, window_minutes=2):
    """
    Add a jet to the master list if no duplicate exists within the time window.

    Parameters
    ----------
    entries : list of dict
        The current master list (modified in place if added).
    data_dict : dict
        The data dictionary returned by jet_reversal_check().
    crossing_time : datetime.datetime
        The user-input crossing time.
    params : dict
        The sidebar parameters used for this run (probe, dt, data_rate, etc.).
    window_minutes : int or float
        Deduplication window in minutes (default 2).

    Returns
    -------
    tuple of (bool, dict or None)
        (True, None) if the jet was added as new.
        (False, existing_entry) if a duplicate was found.
    """
    jet_time_str = str(data_dict.get("jet_time", crossing_time))

    existing = find_nearby_jet(entries, jet_time_str, window_minutes=window_minutes)
    if existing is not None:
        return False, existing

    # Build a serializable entry with rounded jet_time (nearest second)
    jet_time_rounded = _round_time_to_seconds(jet_time_str)
    crossing_rounded = _round_time_to_seconds(str(crossing_time))

    entry = {
        "jet_time": jet_time_rounded,
        "crossing_time": crossing_rounded,
        "added_at": datetime.datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Add sidebar parameters
    for key in ["mms_probe", "dt", "jet_len", "data_rate", "level", "coord_type", "time_clip",
                "tsy_model", "recon_models", "omni_level", "m_p", "dr", "limits"]:
        if key in params:
            entry[key] = params[key]

    # Add all data_dict fields with intelligent rounding
    for key, value in data_dict.items():
        rounded = _round_for_storage(key, value)
        entry[f"data_{key}"] = _to_serializable(rounded)

    entries.append(entry)
    return True, None


def _round_time_to_seconds(time_str):
    """
    Truncate a time string to the nearest second.

    Parameters
    ----------
    time_str : str
        A datetime string, possibly with sub-second precision.

    Returns
    -------
    str
        The time string truncated to seconds (YYYY-MM-DD HH:MM:SS or with +00:00).
    """
    try:
        dt = _parse_time(time_str)
        dt = dt.replace(microsecond=0)
        # Return in a clean format
        if dt.tzinfo is not None:
            return dt.strftime("%Y-%m-%d %H:%M:%S%z")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(time_str)


# Rounding rules by field name pattern.
# Keys are checked via substring match; order matters (first match wins).
# "decimals" = number of decimal places for floats.
# "integer" = cast to int.
# "time" = truncate to nearest second.
_ROUNDING_RULES = [
    # Timestamps → nearest second
    (["Date", "jet_time"], "time"),
    # Indices → integer
    (["ind_min", "ind_max", "ind_jet"], "integer"),
    # Probe, detection → integer
    (["Probe", "jet_detection"], "integer"),
    # Positions in RE → 3 decimal places
    (["x_gsm", "y_gsm", "z_gsm", "r_spc"], 3),
    # Angles in degrees → 2 decimal places
    (["angle_", "imf_clock_angle", "cone_angle"], 2),
    # B-field components in nT → 3 decimal places
    (["b_lmn_vec", "b_imf", "b_msh"], 3),
    # Number density in cm^-3 → 2 decimal places
    (["np_msp", "np_msh", "np_imf"], 2),
    # Velocity in km/s → 2 decimal places
    (["vp_lmn_vec", "v_imf"], 2),
    # Temperatures (large values in K) → integer
    (["tp_para", "tp_perp"], "integer"),
    # Plasma beta → 3 decimal places
    (["beta_"], 3),
    # Dynamic pressure → 2 decimal places
    (["p_dyn"], 2),
    # Reconnection distance → 3 decimal places
    (["r_rc"], 3),
    # Dipole tilt → 2 decimal places
    (["dipole"], 2),
]


def _round_for_storage(key, value):
    """
    Apply intelligent rounding based on the physical meaning of a field.

    Parameters
    ----------
    key : str
        The field name from data_dict.
    value : any
        The value to round.

    Returns
    -------
    any
        The rounded value, or the original if no rule matches or if
        the value is not numeric.
    """
    import numpy as np

    for patterns, rule in _ROUNDING_RULES:
        if any(p in key for p in patterns):
            if rule == "time":
                return _round_time_to_seconds(str(value))
            elif rule == "integer":
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return value
            elif isinstance(rule, int):
                # Decimal places
                try:
                    return round(float(value), rule)
                except (ValueError, TypeError):
                    return value

    # Default: if it's a float, round to 3 decimal places
    if isinstance(value, (float, np.floating)):
        return round(float(value), 3)

    return value


def _to_serializable(value):
    """
    Convert a value to a JSON-serializable type.

    Parameters
    ----------
    value : any
        The value to convert.

    Returns
    -------
    any
        A JSON-serializable version of the value.
    """
    import numpy as np
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime.datetime,)):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def delete_jets(entries, indices):
    """
    Remove jets from the master list by index.

    Parameters
    ----------
    entries : list of dict
        The current master list (modified in place).
    indices : list of int
        Sorted list of indices to remove.

    Returns
    -------
    list of dict
        The modified master list with entries removed.
    """
    # Remove in reverse order to preserve indices
    for idx in sorted(indices, reverse=True):
        if 0 <= idx < len(entries):
            entries.pop(idx)
    return entries


def export_to_csv(entries, path="master_jets.csv"):
    """
    Export the master jet list to a CSV file.

    Parameters
    ----------
    entries : list of dict
        The master list entries.
    path : str
        Output file path.

    Returns
    -------
    bytes
        The CSV content as bytes for download.
    """
    if not entries:
        return b""
    df = pd.DataFrame(entries)
    return df.to_csv(index=False).encode("utf-8")


def export_to_json(entries):
    """
    Export the master jet list to JSON bytes.

    Parameters
    ----------
    entries : list of dict
        The master list entries.

    Returns
    -------
    bytes
        The JSON content as bytes for download.
    """
    return json.dumps(entries, indent=2, default=str).encode("utf-8")


def export_to_pickle(entries):
    """
    Export the master jet list to pickle bytes.

    Parameters
    ----------
    entries : list of dict
        The master list entries.

    Returns
    -------
    bytes
        The pickled content as bytes for download.
    """
    return pickle.dumps(entries)


def master_list_to_stats_csv(entries, time_start=None, time_end=None):
    """
    Convert master list entries to a CSV string compatible with the statistics plots.

    Filters entries to the given time range and reconstructs the wide-format CSV
    that generate_statistics_plots() expects.

    Parameters
    ----------
    entries : list of dict
        The master list entries.
    time_start : datetime.datetime, optional
        Start of time range filter.
    time_end : datetime.datetime, optional
        End of time range filter.

    Returns
    -------
    str or None
        Path to a temporary CSV file, or None if no matching entries.
    """
    if not entries:
        return None

    filtered = entries
    if time_start is not None:
        time_start = _parse_time(time_start)
        filtered = [
            e for e in filtered
            if _parse_time(e.get("jet_time", e.get("crossing_time"))) >= time_start
        ]
    if time_end is not None:
        time_end = _parse_time(time_end)
        filtered = [
            e for e in filtered
            if _parse_time(e.get("jet_time", e.get("crossing_time"))) <= time_end
        ]

    if not filtered:
        return None

    df = pd.DataFrame(filtered)

    # Rename columns to remove 'data_' prefix for compatibility with stats plots
    rename_map = {c: c[5:] for c in df.columns if c.startswith("data_")}
    df.rename(columns=rename_map, inplace=True)
    
    # Duplicate the dataframe 4 times to populate the standard 2x2 plot grid
    models = ["shear", "rx_en", "va_cs", "bisection"]
    
    # We map the standard model names to the exact keys output by the ridge finder.
    # We stripped "data_" earlier, so the column will be "r_rc_Shear", "r_rc_Reconnection Energy", etc.
    model_mapping = {
        "shear": "r_rc_Shear",
        "bisection": "r_rc_Bisection Field",
        "rx_en": "r_rc_Reconnection Energy",
        "va_cs": "r_rc_Exhaust Velocity",
    }
    
    df_list = []
    for model in models:
        df_copy = df.copy()
        df_copy["method_used"] = model
        
        # Check if the theoretical R_rc was computed and saved for this model
        target_col = model_mapping[model]
        if target_col in df.columns:
            df_copy["r_rc"] = df[target_col]
        else:
            df_copy["r_rc"] = np.nan
            
        df_list.append(df_copy)
        
    return pd.concat(df_list, ignore_index=True)
