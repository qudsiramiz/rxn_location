# Reconnection line location

## Introduction
Statistical comparison between X-line location predicted by various models for dayside terrestrial
magnetopause. The models used are:
1. [Maximum Shear Model](https://doi.org/10.1029/2007JA012270)
2. [ Maximum Reconnection Energy](https://doi.org/10.1063/1.4811467)
3. [Maximum Exhaust Velocity](https://doi.org/10.1063/1.2795630)
4. [Maximum Bisection Field](https://doi.org/10.1029/2002JA009381)

The MMS data used in this study is from [MMS](https://lasp.colorado.edu/mms/sdc/public/) mission.
The OMNI data is from [OMNIWeb](https://omniweb.gsfc.nasa.gov/).

The repository is archived using Zenodo with the following DOI:
[![DOI](https://zenodo.org/badge/669151405.svg)](https://zenodo.org/badge/latestdoi/669151405)

## Description
This repository contains the code and data used to generate figures in the paper "Statistical
comparison of various dayside magnetopause reconnection X-line prediction models" by Ramiz A. Qudsi,
Brian Walsh, Jeff Broll, Emil Atz, Stein Haaland.

## Code
The code is written for Python 3.11 or later and has the following dependencies:

```
[tool.poetry.dependencies]
python = ">=3.11,<4.0"
spacepy = "^0.7.0"
pyspedas = "^2.1.3"
tabulate = "^0.9.0"
trjtrypy = "^0.0.0"
joblib = "^1.3.1"
ipython = "^9.15.0"
h5py = "^3.9.0"
scikit-image = "^0.26.0"
more-itertools = "^11.1.0"
seaborn = "^0.13.2"
matplotlib = "^3.11.0"

[tool.poetry.dev-dependencies]

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```
### Running the code
The code can be run using python or ipython. Since the code has a lot of dependencies, it is
recommended to use a virtual environment.
Since the package uses SpacePy and PySPEDAS, installation of which can be a bit tricky becasue of
internal dependencies, we strongly recommend using Poetry to install the dependencies and run the
code.

Poetry can be installed using pip:
```
pip install poetry
```
Once Poetry is installed, the dependencies can be installed using:
```
poetry install
```
This assumes that the code has been cloned to the local machine and that ```poetry.lock``` and
```pyproject.toml``` are present in the root directory of the repository.

Once the dependencies are installed, start the virtual environment using:
```
poetry shell
```

### Interactive GUI

The easiest way to use `rxn_location` is via the interactive Streamlit graphical user interface. This interface allows you to run Jet Reversal checks, visualize 3D reconnection models, run automated statistical batch modes, and generate statistical plots dynamically.

**Recent GUI Features include:**
- **Master Jet List**: Persistent JSON storage of detected jets with automatic 2-minute deduplication across sessions. Now automatically fetches and saves contextual Solar Wind parameters (B_IMF, V_IMF, Np, Tp, Sym-H, Clock Angle, P_dyn).
- **Interactive Data Table**: View, sort, manually prune, and export (CSV/JSON/Pickle) your master jet list. Features high-quality Unicode formatting for variables.
- **Duplicate Jet Dialog**: Safety checks when generating models to prevent processing the same jet multiple times.
- **Parameter Presets**: Save and load your favorite sidebar configurations.
- **Data Cache Dashboard**: Monitor and clean up the local PySPEDAS data cache directly from the sidebar.
- **Dynamic Plot Filtering**: Filter statistics by IMF Bz, dynamic pressure, and shear angle before plotting.
- **Quick Re-run & Auto-Run**: Instantly load parameters from a previously saved jet into the dashboard ("Load into Dashboard"), or use the new "Load & Run Models" button to auto-execute the entire pipeline in one click.
- **Batch Processing from File Upload**: In addition to time-range and target-count batch processing, "Statistics Mode" now accepts uploaded `.csv` or `.txt` files containing custom lists of timestamps. It leverages robust parsing to seamlessly run the jet reversal check and models on every timestamp provided.

To launch the GUI, run the following command from the root directory:
```bash
streamlit run src/rxn_location/app.py
```

### Command Line Scripts

In order to check for jet location in MMS data via command line, use the following command:
```
python -m jet_reversal_check.py
```

#### Batch Processing via Command Line
If you prefer not to use the GUI, you can completely automate the entire pipeline (Jet Reversal Check -> Reconnection Models -> Master Jet List logging) using the included batch statistics command-line script.

1. **Prepare your input file**: Create a `.txt` or `.csv` file (e.g. `times_list.txt`) containing the timestamps you want to check, one per line.
    ```text
    2015-09-02 16:45:00
    2015-09-02 17:30:00
    2015-10-16 13:07:00
    ```
2. **Run the script**: Run `batch_statistics_cli.py` from the `examples/` folder and pass your input file.
    ```bash
    python examples/batch_statistics_cli.py --input times_list.txt --probe 3 --format html --outdir ./my_batch_figures
    ```

**Available Options:**
- `-i`, `--input`: (Required) Path to your `.txt` or `.csv` list of times.
- `--probe`: MMS probe number (1-4). Defaults to `3`.
- `--tsy_model`: Tsyganenko model to use. Defaults to `T96`.
- `--data_rate`: MMS data rate (`fast` or `brst`). Defaults to `fast`.
- `--format`: Format for the saved plots (`html` for interactive Plotly, or `pdf`/`png` for static Matplotlib). Defaults to `html`.
- `--outdir`: Directory to save the generated plots and CSV. Defaults to `./figures`.
- `--log-level`: Set the terminal output verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). Defaults to `INFO`.

When run, the script will output its default configurations via a neat table, process each time, print the status, fetch Solar Wind parameters, generate the required plots, and intelligently skip logging duplicate jets to your persistent `~/.rxn_location_master_jets.json` file.

NOTE: Some times, for whatever reason, the MMS data is not downloaded properly by PySPEDAS. In that
case, please run the following command:
```
python -m spd_brst_test.py
```

This will create two figures in the ```figures``` directory. One figure is from FPI data and another
frpm FGM data. If those two figures are created properly, that means the MMS data is downloaded and
you should be able to run the ```jet_reversal_check.py``` code.

Please refer to the code documentation for more details on the code.

Next step would be to produce the reconnection line location figures. This can be done using the 
following code/command:
```
python -m rx_code.py
```

In order to compute the statistics from the figures, use the following code/command:
```
python -m rc_stats.py
```

Other figures are plotted using Seaborn package and is slightly more involved. You can use the
following command to generate the figures:
```
python -m seaborn_plots.py
```
However, sometimes, that gives error depending on some other conditions. Currently we are working on
fixing that issue.

## Data
All the data used to generate the figures in the paper are available in the ```data/study_data```
directory.

# Contact
If you have any questions, please contact [Ramiz A. Qudsi](https://www.qudsiramiz.space/) at
qudsiramiz@gmail.com
