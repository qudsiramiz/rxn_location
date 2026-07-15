import pytest
from rxn_location.jet_reversal_check_function import jet_reversal_check

def test_jet_reversal_check_import():
    # Basic import test and verify signature
    assert callable(jet_reversal_check)
