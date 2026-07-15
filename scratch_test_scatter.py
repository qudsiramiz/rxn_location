from src.rxn_location.app_stats_plots import generate_statistics_plots

csv_path = "src/rxn_location/reconnection_stats_2015-09-02_164500_2jets_10m.csv"
fig, err = generate_statistics_plots(csv_path, dark_mode=False, selected_plots=['Scatter Plots'], x_var='b_imf_z', y_var='b_imf_y')
print("Error:", err)
print("Figures:", fig.keys())
