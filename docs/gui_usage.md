# Interactive GUI Usage

`rxn_location` comes with a powerful interactive Streamlit application that allows you to analyze MMS spacecraft data, run Jet Reversal checks, visualize reconnection models, and generate statistical plots—all from an easy-to-use web interface.

## Launching the App

> [!IMPORTANT]
> **Virtual Environment Highly Recommended**
> Before running the GUI, make sure you have activated the virtual environment where you installed the package. If you used `venv`, activate it via `source venv/bin/activate` (or `venv\Scripts\activate` on Windows). If you used Poetry, start your shell via `poetry shell`. If you don't activate the environment, the `rxn-location-gui` command will not be found.

To launch the graphical user interface, simply navigate to the root of your project directory and run the following command in your terminal:

```bash
rxn-location-gui
```

This will automatically open the interactive GUI in your default web browser.

---

## The Layout

The application is divided into a **Sidebar** on the left (for global settings) and **Four Main Tabs** on the right for interacting with your data and results.

### Global Settings (Sidebar)
The sidebar contains global controls that apply across all tabs:

| Parameter | Description | Typical Range | Default |
| :--- | :--- | :--- | :--- |
| **Dark Mode** | Toggle between dark mode and light mode plots. | N/A | Dark |
| **Crossing Time** | The UTC timestamp to center the analysis around. | Valid MMS operational dates | `2015-09-02 16:45:00` |
| **MMS Probe** | Selects which of the 4 MMS spacecraft to source telemetry from. | `1` to `4` | `3` |
| **Data Rate** | The telemetry data rate from MMS instruments. | `brst` (burst), `fast`, `srvy` (survey) | `brst` |
| **Data Level** | Data processing level. | `l2` | `l2` |
| **Coordinate Type** | Spatial coordinate system used for vectors. | `gsm`, `gse` | `gsm` |
| **Tsyganenko Model**| The background empirical magnetic field model to use. | `t89`, `t96`, `t01`, `ts04` | `t96` |

---

### Tab 1: Controls
This tab is your starting point for running single-event analyses. 

#### 1. Jet Reversal Parameters
Configure the physics parameters for the Jet detection algorithm.

| Parameter | Description | Typical Range | Default |
| :--- | :--- | :--- | :--- |
| **Time Window (dt)** | The time radius (in seconds) around the crossing time to search for jets. | `60` - `600` | `300` |
| **Jet Length Threshold** | Minimum consecutive data points exceeding threshold to qualify as a valid jet. | `1` - `10` | `3` |
| **Find Next Jet Time Delta**| When clicking "Find Next Jet", how far forward in time (minutes) to jump for each check. | `5` - `30` | `10` |
| **Max Attempts for Next Jet**| The maximum number of consecutive checks to make before giving up on finding the next jet. | `3` - `20` | `5` |

#### 2. Reconnection Models Parameters
Configure the spatial bounds and constants for the 3D reconnection distance plotting.

| Parameter | Description | Typical Range | Default |
| :--- | :--- | :--- | :--- |
| **OMNI Level** | Cadence for solar wind data. | `hro_1min`, `hro_5min` | `hro_1min` |
| **Proton Mass Multiple (m_p)** | A scaling factor applied to the proton mass in model calculations. | `0.5` - `2.0` | `1.0` |
| **Grid Resolution (dr)** | Spatial resolution of the plotting grid in Earth Radii ($R_E$). Smaller is finer but slower. | `0.1` - `1.0` | `0.5` |
| **Grid Limits (±)** | Maximum physical extent of the X and Y bounds for 3D generation (in $R_E$). | `10.0` - `40.0` | `20.0` |

---

### Tab 2: Jet Reversal Plot
When you run a Jet Reversal Check, the resulting Plotly figure is displayed here.
- It is fully interactive: you can zoom, pan, and hover over data points to see exact values.
- A dropdown menu at the bottom allows you to view the plots for any previously tested time from your current session history.

---

### Tab 3: Reconnection Models
This tab allows you to visualize where reconnection occurs based on several models (Shear, Reconnection Energy, Exhaust Velocity, Bisection).

- **Reconnection Models Selection:** Check/uncheck which models you want to plot.
- The generated interactive 3D plots will be displayed here after you run the models from the Controls tab or via Statistics Mode.

---

### Tab 4: Statistics Results
This tab is designed for batch processing and generating aggregated statistical plots using Seaborn and Matplotlib.

#### Statistics Mode Configuration
!!! tip "Automated Loop"
    Statistics mode lets you automatically step through a range of time, testing for jets and computing models at every interval, then saving all the parameters to a CSV file.

| Parameter | Description | Typical Range | Default |
| :--- | :--- | :--- | :--- |
| **Stop Condition** | Choose whether to stop at a specific UTC End Time or after finding a Target Number of Jets. | `End Time`, `Target Number` | `End Time` |
| **End Time / Target Jets** | The respective goal based on the Stop Condition selected above. | N/A | `10` jets |
| **Time Delta (minutes)** | How many minutes to step forward after each automated check. | `5` - `60` | `15` |
| **Output CSV Name** | Filename for saving the batch loop output. Automatically generated. | N/A | dynamic |

When you click **Run Statistics Mode**, the app will iterate forward in time. For every valid jet detection, it runs the Reconnection Models, gathers all physical parameters (like `r_rc`, MMS position, IMF data, and jet data), and appends them to your CSV file. **During the loop, you will see a live preview of the latest Jet Reversal plot.**

#### Generating Figures
Once you have generated (or uploaded) a CSV file, you can generate aggregated plots:
1. **Plots to Generate:** Choose from *Histograms*, *KDE Plots*, *2D Histograms*, *Scatter Plots*, or the *MMS Location Scatter Plot*.
2. **Dynamic Variables:** If you select KDE, Scatter, or 2D Histograms, use dropdown menus to choose precisely which variables you want to plot on the X and Y axes (e.g., IMF Bz vs Reconnection Distance).
3. Click **Generate Selected Figures** to view your publication-ready plots!
