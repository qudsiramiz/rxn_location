import pytest
import numpy as np
from rxn_location.rx_model_funcs import get_shear, get_rxben, get_bis

def test_get_shear():
    b_vec_1 = [1, 0, 0]
    b_vec_2 = [0, 1, 0]
    # Shear angle between (1,0,0) and (0,1,0) should be 90 degrees
    shear = get_shear(b_vec_1, b_vec_2, angle_unit="degrees")
    assert np.isclose(shear, 90.0)

def test_get_rxben():
    b_vec_1 = [1, 0, 0]
    b_vec_2 = [0, 1, 0]
    rx_en = get_rxben(b_vec_1, b_vec_2)
    assert isinstance(rx_en, float)

def test_get_bis():
    b_vec_1 = [1, 0, 0]
    b_vec_2 = [0, 1, 0]
    bisec_msp, bisec_msh = get_bis(b_vec_1, b_vec_2)
    assert isinstance(bisec_msp, (float, np.floating))
    assert isinstance(bisec_msh, (float, np.floating))
