# Command Line Interface (CLI) Usage

While `rxn_location` features a rich interactive graphical interface (`rxn-location-gui`), it also provides a powerful Command Line Interface (CLI) called `rxn-batch`. This tool allows you to automate the entire data processing pipeline—from downloading MMS data and detecting jets to running 3D reconnection models and logging the results into your Master Jet List.

## Basic Usage

> [!IMPORTANT]
> **Virtual Environment Highly Recommended**
> Before running any commands, make sure you have activated the virtual environment where you installed the package. If you used `venv`, activate it via `source venv/bin/activate` (or `venv\Scripts\activate` on Windows). If you used Poetry, start your shell via `poetry shell`. If you don't activate the environment, the `rxn-batch` command will not be found.

The primary requirement for `rxn-batch` is an input file containing a list of target timestamps you wish to process.

```bash
rxn-batch --input data/potential_jet_reconnection_times.txt
```

The script reads each timestamp line by line, runs the jet reversal checks, computes the reconnection models, fetches solar wind parameters, generates plots, and logs the valid events to `~/.rxn_location_master_jets.json`.

## Configuration Files

For complex runs with many arguments, you can store your configuration in a text file and pass it to the CLI using the `@` prefix.

1. Create a file named `my_config.txt`:
```text
--input
data/potential_jet_reconnection_times.txt
--probe
1
--outdir
./batch_run_results
--format
png
--plot-seaborn
```

https://www.qudsiramiz.space/rxn_location/latest/cli_usage/sample_files/my_config.txt
> [!NOTE]
> A comprehensive example configuration file containing all possible inputs and their descriptions is available at [sample_files/my_config.txt](../sample_files/my_config.txt).

2. Run the tool with the configuration file:
```bash
rxn-batch @sample_files/my_config.txt
```

## Common Examples

### 1. Change the Target Probe and Output Directory
If you want to look for jets using MMS Probe 2 and save the output figures to a specific directory:
```bash
rxn-batch -i times_list.txt --probe 2 --outdir ./probe2_results
```

### 2. Static Plotting for Publications
By default, the script generates interactive HTML plots using Plotly. For publications, you might prefer static PNG or PDF images rendered by Matplotlib:
```bash
rxn-batch -i times_list.txt --format png --plot_engine matplotlib
```

### 3. Generate Statistical Seaborn Plots Automatically
If you've collected a large list of jets in your master list, you can have the batch script automatically generate all the statistical Seaborn distributions at the very end of its run:
```bash
rxn-batch -i times_list.txt --plot-seaborn
```

### 4. Adjust the Reconnection Model Grid Settings
If you want a higher resolution 3D model (e.g. `dr = 0.1` $R_E$) covering a larger spatial extent (`limits = 30` $R_E$):
```bash
rxn-batch -i times_list.txt --dr 0.1 --limits 30
```

## Full List of Arguments

You can always view the full list of available arguments by running:
```bash
rxn-batch --help
```

| Argument | Description | Default |
| :--- | :--- | :--- |
| `-i`, `--input` | **(Required)** Path to input `.txt` or `.csv` file with a list of times. | None |
| `--probe` | MMS probe number (1-4). | `3` |
| `--tsy_model` | Tsyganenko model to use. | `T96` |
| `--data_rate` | MMS data rate (`fast` or `brst`). | `fast` |
| `--format` | Format for saved plots (`html`, `pdf`, `png`). | `html` |
| `--outdir` | Output directory for plots. | `./figures` |
| `--csv_name` | Name of output stats CSV (if any). | dynamic |
| `--dt` | Time window around crossing time to search for jets (s). | `300` |
| `--jet_len` | Minimum number of data points for a valid jet. | `3` |
| `--m_p` | Magnetopause standoff distance scaling. | `0.5` |
| `--dr` | Spatial resolution for 3D model grid ($R_E$). | `0.25` |
| `--limits` | X, Y, Z boundaries for 3D grid ($R_E$). | `20` |
| `--verbosity` | Verbosity level (0-3). 0: silent, 1: important only, 2: standard, 3: all. | `2` |
| `--t_delta` | Minutes to shift the crossing time on each retry when searching for jets. | `2` |
| `--max_retries` | Maximum number of time-shifted retries when initial jet check fails. | `5` |
| `--plot-seaborn` | Generate final statistical Seaborn joint-plots from the Master List after the batch run finishes. | Disabled |
| `--plot_engine` | Engine used to plot the jet reversal check (`plotly` or `matplotlib`). | `plotly` |

