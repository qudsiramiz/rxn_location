# rxn_location
Statistical comparison between X-line location predicted by various models for dayside terrestrial magnetopause

## Description
This repository contains the code and data used to generate figures in the paper "Statistical
comparison of various dayside magnetopause reconnection X-line prediction models" by Ramiz A. Qudsi,
Brian Walsh, Jeff Broll, Emil Atz, Stein Haaland.

## Code
The code is written in Python 3.10 and has the following dependencies:

```
[tool.poetry.dependencies]
python = ">=3.10, <3.11"
spacepy = "^0.4.1"
pyspedas = "^1.4.40"
tabulate = "^0.9.0"
trjtrypy = "^0.0.0"
joblib = "^1.3.1"
ipython = "^8.14.0"
h5py = "^3.9.0"
scikit-image = "^0.21.0"
more-itertools = "^9.1.0"
seaborn = "0.12.1"
matplotlib = "3.5.2"

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

In order to check for jet location in MMS data, use the following command:
```
python -m jet_reversal_check.py
```

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
If you have any questions, please contact [Ramiz A. Qudsi](https://www.qudsiramiz.space/) at qudsiramiz@gmail.com
