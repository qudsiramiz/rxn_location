import datetime
import pytz
import pandas as pd
import jet_reversal_check_function as jrcf

df_crossings = pd.read_csv("../data/brst_intervals.csv", index_col=False)
df_crossings.set_index("start_time", inplace=True)

crossing_time_raw = df_crossings.index[1004]
print(f"Testing crossing_time = {crossing_time_raw}")
crossing_time = datetime.datetime.strptime(crossing_time_raw.split("+")[0], "%Y-%m-%d %H:%M:%S")
crossing_time = crossing_time.replace(tzinfo=pytz.utc)

jrcf.jet_reversal_check(crossing_time=crossing_time)
