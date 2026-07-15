import pyspedas as spd
import numpy as np

time = np.array([1, 2, 3])
y = np.array([10, 20, 30])
spd.store_data('var1', data={'x': time, 'y': y})
spd.store_data('var2', data={'x': time, 'y': y*2})

spd.store_data('pseudo', data=['var1', 'var2'])

meta = spd.get_data('pseudo', metadata=True)
print("pseudo meta:", meta)
print("pseudo data:", spd.get_data('pseudo'))

meta_var1 = spd.get_data('var1', metadata=True)
print("var1 meta:", meta_var1)
print("var1 data:", spd.get_data('var1'))

