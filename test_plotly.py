import pyspedas as spd
import pandas as pd
import numpy as np

spd.store_data('test', data={'x': [1,2,3], 'y': [4,5,6]})
spd.options('test', 'ytitle', 'Test Title')

meta = spd.get_data('test', metadata=True)
print(meta)

# get options specifically
from pytplot import data_quants
if 'test' in data_quants:
    print(data_quants['test'].attrs['plot_options'])
