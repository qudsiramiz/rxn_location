# rxn_location package initialization

from importlib import import_module

__version__ = "0.3.0"

__all__ = ["rx_model", "get_sw_params", "jet_reversal_check", "plot_hist", "__version__"]

def __getattr__(name):
    if name in ("rx_model", "get_sw_params"):
        return getattr(import_module(".rx_model_funcs", __name__), name)
    if name == "jet_reversal_check":
        return getattr(import_module(".jet_reversal_check_function", __name__), name)
    if name == "plot_hist":
        return getattr(import_module(".rc_stats_fncs", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return __all__
