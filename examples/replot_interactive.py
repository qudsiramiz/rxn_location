import pickle
from rxn_location import rx_model_funcs as rmf


def replot():
    # Load the cached inputs
    with open("data/rx_d/plot_inputs_mms3_0.pkl", "rb") as f:
        figure_inputs = pickle.load(f)

    print("Generating interactive full-screen Plotly plot...")
    rmf.ridge_finder_multiple_interactive(**figure_inputs)
    print("Done! Check the interactive_figures/ directory.")


if __name__ == "__main__":
    replot()
