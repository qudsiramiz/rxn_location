import traceback
import sys
from rxn_location.app_stats_plots import generate_statistics_plots

csv_path = "data/study_data/reconnection_stats.csv"

try:
    figures, err = generate_statistics_plots(
        csv_path,
        dark_mode=False,
        selected_plots=["KDE Plots"],
        x_var="b_imf_y",
        y_var="b_imf_x"
    )
    print("Success!", figures, err)
except Exception as e:
    print("Caught exception:")
    traceback.print_exc()
