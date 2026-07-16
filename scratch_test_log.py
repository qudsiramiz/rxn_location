import logging
from rxn_location.jet_reversal_check_function import jet_reversal_check

print("Loggers active:")
for name in logging.root.manager.loggerDict:
    if 'sped' in name.lower() or 'tplot' in name.lower():
        print(name)
