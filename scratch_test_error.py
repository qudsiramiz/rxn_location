import sys
import datetime
from rxn_location.jet_reversal_check_function import jet_reversal_check
import pytz

c_time = datetime.datetime(2015, 9, 2, 16, 45, 0, tzinfo=pytz.utc)

try:
    fig, s1, det = jet_reversal_check(
        c_time,
        probe="3",
        data_rate="fast",
        dt=300,
        jet_len=3,
    )
except Exception as e:
    import traceback
    traceback.print_exc()
