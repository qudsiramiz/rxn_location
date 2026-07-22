# Reconnection X-line location (`rxn_location`)

## Introduction
`rxn_location` is a Python package for computing the model predicted X-line location and analyzing them at the dayside terrestrial magnetopause. It provides tools to analyze individual reconnection jets via an interactive GUI, as well as CLI tools for automated, large-scale statistical batch processing.

The package predicts the X-line location based on four primary models:

1. [Maximum Shear Model](https://doi.org/10.1029/2007JA012270)
2. [Maximum Reconnection Energy](https://doi.org/10.1063/1.4811467)
3. [Maximum Exhaust Velocity](https://doi.org/10.1063/1.2795630)
4. [Maximum Bisection Field](https://doi.org/10.1029/2002JA009381)

The MMS data used by this software is automatically fetched from the [MMS](https://lasp.colorado.edu/mms/sdc/public/) mission, and OMNI solar wind data is fetched from [OMNIWeb](https://omniweb.gsfc.nasa.gov/).

The repository is archived using Zenodo with the following DOI:
[![DOI](https://zenodo.org/badge/669151405.svg)](https://zenodo.org/badge/latestdoi/669151405)

## Description
This package originally started as the repository for generating the figures in the paper ["Statistical comparison of various dayside magnetopause reconnection X-line prediction models"](https://doi.org/10.1029/2023JA031644) (by Ramiz A. Qudsi, Brian Walsh, Jeff Broll, Emil Atz, Stein Haaland). It has since evolved into a software tool suite for both individual jet event analysis and aggregated statistical analysis.

## Code
The code is written for Python 3.11 or later. All dependencies are managed automatically via `pyproject.toml` and will be installed when you install the package.

Key scientific and visualization libraries used by `rxn_location` include:

- `spacepy` & `pyspedas` (for MMS and OMNI data retrieval and physics algorithms)
- `streamlit` (for the interactive graphical interface)
- `plotly`, `seaborn`, & `matplotlib` (for interactive and statistical visualizations)
- `scikit-image` (for ridge-finding models)
- `joblib` (for parallelized batch processing)

### Installation & Setup
Since the code has many dependencies (including SpacePy and PySPEDAS, which can sometimes be tricky to install), we highly recommend using a virtual environment. You can install the package directly from PyPI (recommended), from GitHub, or by cloning the repository to install it locally.

#### Option 1: Install from PyPI (Recommended)
The easiest way to get started and use the `rxn-location-gui` and `rxn-batch` tools is to install the latest stable release directly from PyPI into a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install rxn-location
```

#### Option 1b: Install from PyPi using poetry (Highly Recommended)
[Poetry](https://python-poetry.org/) provides a great way to manage dependencies and is often better at installing packages with complex dependencies than pip.
Assuming you have poetry installed, you can install the package by running the following command (in a folder of your choice):
```bash
poetry init --no-interaction
poetry add rxn-location
```

#### Option 2: Install Directly from GitHub
If you want the absolute latest development version without modifying the source code, you can install directly from the GitHub repository:
```bash
pip install git+https://github.com/qudsiramiz/rxn_location.git
```

#### Option 3: Clone and Install Locally (Standard Python)
If you want to explore the source code, run examples, or contribute, you should clone the repository first:
```bash
git clone https://github.com/qudsiramiz/rxn_location.git
cd rxn_location

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in editable mode
pip install -e .
```

#### Option 4: Clone and Install Locally (Poetry)
If you prefer using [Poetry](https://python-poetry.org/) for dependency management:
```bash
git clone https://github.com/qudsiramiz/rxn_location.git
cd rxn_location

# Install Poetry (if you don't have it)
pip install poetry

# Install dependencies and start the environment
poetry install
poetry shell
```

### Interactive GUI

The simplest way to use `rxn_location` is via the interactive Streamlit graphical user interface. This interface allows you to run Jet Reversal checks, visualize reconnection models, run automated statistical batch modes, and generate statistical plots dynamically.

**Recent GUI Features include:**

- **Master Jet List**: Persistent JSON storage of detected jets with automatic 2-minute deduplication across sessions. Now automatically fetches and saves contextual Solar Wind parameters (B_IMF, V_IMF, Np, Tp, Sym-H, Clock Angle, P_dyn).
- **Interactive Data Table**: View, sort, manually prune, and export (CSV/JSON/Pickle) your master jet list. Features high-quality Unicode formatting for variables.
- **Duplicate Jet Dialog**: Safety checks when generating models to prevent processing the same jet multiple times.
- **Parameter Presets**: Save and load your sidebar configurations.
- **Data Cache Dashboard**: Monitor and clean up the local PySPEDAS data cache directly from the sidebar.
- **Dynamic Plot Filtering**: Filter statistics by IMF Bz, dynamic pressure, and shear angle before plotting.
- **Advanced Plotting Engines**: Toggle between `Plotly (Interactive)` and `Matplotlib (Static)` engines dynamically when generating statistical plots.
- **Statistical Plots**: Added support for interactive 2D Density Histograms (Hexbins) in Plotly, and Box Plots & Violin Plots in Matplotlib.
- **Batch Processing from File Upload**: In addition to time-range and target-count batch processing, "Statistics Mode" now accepts uploaded `.csv` or `.txt` files containing custom lists of timestamps. It leverages robust parsing to seamlessly run the jet reversal check and models on every timestamp provided.

To launch the GUI, run the following command from the root directory:
```bash
rxn-location-gui
```

In order to check for jet locations and process statistics from the command line, we recommend using the new automated batch script:

1. **Prepare your input file**: Create a `.txt` or `.csv` file (e.g. `times_list.txt`) containing the timestamps you want to check, one per line.
    ```text
    2015-09-02 16:45:00
    2015-09-02 17:30:00
    2015-10-16 13:07:00
    ```
2. **Run the script**: Run `rxn-batch` and pass your input file.
    ```bash
    rxn-batch --input times_list.txt --probe 3 --format html --outdir ./my_batch_figures
    ```

**Available Options:**

- `-i`, `--input`: (Required) Path to your `.txt` or `.csv` list of times.
- `--probe`: MMS probe number (1-4). Defaults to `3`.
- `--tsy_model`: Tsyganenko model to use. Defaults to `T96`.
- `--data_rate`: MMS data rate (`fast` or `brst`). Defaults to `fast`.
- `--format`: Format for the saved plots (`html` for interactive Plotly, or `pdf`/`png` for static Matplotlib). Defaults to `html`.
- `--outdir`: Directory to save the generated plots and CSV. Defaults to `./figures`.
- `--verbosity`: Set the terminal output verbosity (`0` to `3`). Defaults to `2`.

When run, the script will output its default configurations, process each time, fetch Solar Wind parameters, generate the required plots, and intelligently skip logging duplicate jets to your persistent `~/.rxn_location_master_jets.json` file.

**For more examples and advanced usage (including configuration files), see the [CLI Usage Guide](cli_usage.md).**

NOTE: Sometimes, the MMS data is not downloaded properly by PySPEDAS. In that case, please check your internet connection or use the GUI's Data Cache Dashboard to manage corrupted files.

# Acknowledgement
Development of this package was supported by the NASA grant 80NSSC20K1356 and the successive Supplemental Open Source Software Award.

# Contact
If you have any questions, please contact [Ramiz A. Qudsi](https://www.qudsiramiz.space/) at
[qudsiramiz@gmail.com](mailto:qudsiramiz@gmail.com).
