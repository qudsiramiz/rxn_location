import logging
import sys

_VERBOSITY = 2

def set_verbosity(level):
    """
    Configure global verbosity level:
    3: Everything prints (including PySPEDAS).
    2: Silence PySPEDAS/PyTplot, keep standard app logs.
    1: Silence detailed app logs (only print major milestones/errors).
    0: Silence completely.
    """
    global _VERBOSITY
    _VERBOSITY = level

    if level <= 2:
        logging.getLogger("pyspedas").setLevel(logging.ERROR)
        logging.getLogger("pytplot").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.ERROR)
    else:
        logging.getLogger("pyspedas").setLevel(logging.INFO)
        logging.getLogger("pytplot").setLevel(logging.INFO)
        logging.getLogger().setLevel(logging.INFO)


def vprint(level, message, color=None):
    """
    Print a message if the current verbosity is >= `level`.
    Available colors: red, green, yellow, blue, magenta, cyan, bold
    """
    if _VERBOSITY >= level:
        if color:
            colors = {
                "red": "\033[91m",
                "green": "\033[92m",
                "yellow": "\033[93m",
                "blue": "\033[94m",
                "magenta": "\033[95m",
                "cyan": "\033[96m",
                "bold": "\033[1m"
            }
            reset = "\033[0m"
            c = colors.get(color.lower(), "")
            print(f"{c}{message}{reset}")
            sys.stdout.flush()
        else:
            print(message)
            sys.stdout.flush()
