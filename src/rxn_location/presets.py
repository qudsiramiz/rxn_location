"""
Presets Management Module

Allows users to save and load sidebar parameter configurations as named presets.
Presets are stored in a JSON file at ~/.rxn_location_presets.json.
"""

import json
import os
from pathlib import Path

DEFAULT_PRESETS_PATH = Path(os.path.expanduser("~")) / ".rxn_location_presets.json"


def load_presets(path=None):
    """
    Load all saved presets from disk.

    Parameters
    ----------
    path : str or Path, optional
        Path to the presets JSON file. Defaults to ~/.rxn_location_presets.json.

    Returns
    -------
    dict
        A dictionary mapping preset names to their parameter dictionaries.
    """
    if path is None:
        path = DEFAULT_PRESETS_PATH
    path = Path(path)
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_presets(presets, path=None):
    """
    Save all presets to disk.

    Parameters
    ----------
    presets : dict
        The full presets dictionary to persist.
    path : str or Path, optional
        Path to the presets JSON file.
    """
    if path is None:
        path = DEFAULT_PRESETS_PATH
    path = Path(path)
    try:
        with open(path, "w") as f:
            json.dump(presets, f, indent=2)
    except IOError:
        pass


def save_preset(name, params, path=None):
    """
    Save or update a single named preset.

    Parameters
    ----------
    name : str
        The preset name.
    params : dict
        The parameter dictionary to save.
    path : str or Path, optional
        Path to the presets JSON file.
    """
    presets = load_presets(path)
    presets[name] = params
    save_presets(presets, path)


def delete_preset(name, path=None):
    """
    Delete a named preset.

    Parameters
    ----------
    name : str
        The preset name to remove.
    path : str or Path, optional
        Path to the presets JSON file.

    Returns
    -------
    bool
        True if the preset was found and deleted, False otherwise.
    """
    presets = load_presets(path)
    if name in presets:
        del presets[name]
        save_presets(presets, path)
        return True
    return False


def list_preset_names(path=None):
    """
    List all available preset names.

    Parameters
    ----------
    path : str or Path, optional
        Path to the presets JSON file.

    Returns
    -------
    list of str
        Sorted list of preset names.
    """
    presets = load_presets(path)
    return sorted(presets.keys())


def get_current_params(session_state):
    """
    Extract the current sidebar parameters from Streamlit session state.

    Parameters
    ----------
    session_state : streamlit.session_state
        The current Streamlit session state.

    Returns
    -------
    dict
        A dictionary of all sidebar parameters.
    """
    return {
        "crossing_time_str": session_state.get(
            "crossing_time_str", "2015-09-02 16:45:00"
        ),
        "mms_probe": session_state.get("preset_mms_probe", 3),
        "dt": session_state.get("preset_dt", 300),
        "jet_len": session_state.get("preset_jet_len", 3),
        "data_rate": session_state.get("preset_data_rate", "brst"),
        "level": session_state.get("preset_level", "l2"),
        "coord_type": session_state.get("preset_coord_type", "lmn"),
        "time_clip": session_state.get("preset_time_clip", True),
        "t_delta": session_state.get("preset_t_delta", 10),
        "max_attempts": session_state.get("preset_max_attempts", 5),
        "tsy_model": session_state.get("preset_tsy_model", "t96"),
        "recon_models": session_state.get(
            "preset_recon_models",
            ["shear", "bisection", "reconnection energy", "exhaust velocity"],
        ),
        "omni_level": session_state.get("preset_omni_level", "hro_1min"),
        "m_p": session_state.get("preset_m_p", 1.0),
        "dr": session_state.get("preset_dr", 0.5),
        "limits": session_state.get("preset_limits", 20.0),
    }
