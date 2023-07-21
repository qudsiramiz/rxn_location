import datetime
import importlib
import numpy as np
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import pandas as pd
import pytz

import jet_reversal_check_function as jrcf

importlib.reload(jrcf)

# Read the data from csv files
df_crossings = pd.read_csv("../data/mms_magnetopause_crossings.csv", index_col=False)

# Set the index to the date column
df_crossings.set_index("DateStart", inplace=True)

date_obs = "20230721"


def check_jet_reversal(crossing_time):
    # Convert the crossing time to a datetime object
    crossing_time = datetime.datetime.strptime(crossing_time.split("+")[0],
                                               "%Y-%m-%d %H:%M:%S.%f")
    # Set the timezone to UTC
    crossing_time = crossing_time.replace(tzinfo=pytz.utc)
    print(f"crossing_time = {crossing_time}")
    # Try with "brst" data rate, if that fails then try with "fast"
    inputs = {"crossing_time": crossing_time,
              "dt": 300,
              "probe": 3,
              "jet_len": 3,
              "level": "l2",
              "coord_type": "lmn",
              "data_type": ["dis-moms", "des-moms"],
              "time_clip": True,
              "latest_version": True,
              "date_obs": date_obs,
              "figname": "mms_jet_reversal_check_lmn_mean",
              "fname": f"../data/mms_jet_reversal_times_list_{date_obs}_brst.csv",
              "error_file_log_name": f"../data/mms_jet_reversal_check_err_log_{date_obs}.csv",
              "verbose": True
              }

    # inputs["data_rate"] = "brst"
    # _ = jrcf.jet_reversal_check(**inputs)
    try:
        try:
            inputs["data_rate"] = "brst"
            _ = jrcf.jet_reversal_check(**inputs)
        except Exception:
            # inputs["data_rate"] = "brst"
            # _ = jrcf.jet_reversal_check(**inputs)
            pass
    except Exception as e:
        print(f"\033[91;31m\n{e} for date {crossing_time}\n\033[0m")
        # Save the crossing time to a file
        # Check if the file exists
        if not os.path.isfile(inputs["error_file_log_name"]):
            # If it doesn"t exist, create it
            with open(inputs["error_file_log_name"], "w") as f:
                f.write("DateStart,Error\n")
                f.write(f"{crossing_time},{e}\n")
        else:
            # If it exists, append to it
            df_added_list = pd.read_csv(inputs["error_file_log_name"], sep=",", index_col=False)
            if not np.any(df_added_list["DateStart"].values == str(crossing_time)):
                with open(inputs["error_file_log_name"], "a") as f:
                    f.write(f"{crossing_time},{e}\n")
                f.close()
        pass

    return None


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


# Ask the user for the minmum and maximum index number
indx_min = int(input("Enter the minimum index number: "))
indx_max = int(input("Enter the maximum index number: "))

if __name__ == "__main__":
    with suppress_stdout_stderr():
        for xx, crossing_time in enumerate(df_crossings.index[indx_min:indx_max], start=indx_min):
            check_jet_reversal(crossing_time)
