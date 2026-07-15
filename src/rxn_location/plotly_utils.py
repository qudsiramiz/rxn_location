import pyspedas as spd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import datetime

import re


def format_latex(text):
    if not text:
        return text
    text = str(text)

    is_latex = "$" in text

    # Remove math mode $
    text = text.replace("$", "")
    # Remove \rm
    text = text.replace("\\rm ", "")
    text = text.replace("\\rm", "")

    # Replace common symbols
    text = text.replace("\\Delta ", "Δ")
    text = text.replace("\\Delta", "Δ")
    text = text.replace("\\parallel ", "∥")
    text = text.replace("\\parallel", "∥")
    text = text.replace("\\perp ", "⟂")
    text = text.replace("\\perp", "⟂")

    if is_latex:
        # Handle subscripts: _{...} or _a
        text = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", text)
        text = re.sub(r"_([a-zA-Z0-9])", r"<sub>\1</sub>", text)

        # Handle superscripts: ^{...} or ^a
        text = re.sub(r"\^\{([^}]+)\}", r"<sup>\1</sup>", text)
        text = re.sub(r"\^([a-zA-Z0-9])", r"<sup>\1</sup>", text)

    return text


def convert_tplot_to_plotly(keys_to_plot, dark_mode=False):
    """
    Converts a list of tplot variables into a Plotly Figure.
    """
    num_plots = len(keys_to_plot)
    fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    layout_updates = {}

    for i, key in enumerate(keys_to_plot):
        meta = spd.get_data(key, metadata=True)
        plot_opts = meta.get("plot_options", {}) if meta else {}

        yaxis_opt = plot_opts.get("yaxis_opt", {})
        zaxis_opt = plot_opts.get("zaxis_opt", {})

        y_title = format_latex(yaxis_opt.get("axis_label", key))
        y_subtitle = format_latex(yaxis_opt.get("axis_subtitle", ""))
        if y_subtitle:
            y_title += f"<br>{y_subtitle}"

        y_type = "log" if yaxis_opt.get("y_axis_type") == "log" else "linear"
        z_type = "log" if zaxis_opt.get("z_axis_type") == "log" else "linear"

        # Calculate vertical center for colorbar/legend
        y_center = 1 - (i + 0.5) / num_plots

        # Determine all variables to plot in this panel (handles pseudo-variables)
        vars_to_plot = plot_opts.get("overplots_mpl", [])
        if not vars_to_plot:
            vars_to_plot = [key]

        legend_names = plot_opts.get("line_opt", {}).get("legend_names", [])
        legend_idx = 0

        for var_name in vars_to_plot:
            data = spd.get_data(var_name)
            if data is None:
                continue

            times = [datetime.datetime.utcfromtimestamp(t) for t in data.times]

            # Check if it's a spectrogram (2D data with v)
            if hasattr(data, "v") or hasattr(data, "v1") or hasattr(data, "v2"):
                v_data = (
                    data.v if hasattr(data, "v") else (data.v1 if hasattr(data, "v1") else data.v2)
                )
                y_data = data.y

                z_title = format_latex(zaxis_opt.get("axis_label", ""))

                # For Plotly heatmap, z is the 2D array, x is time, y is v_data
                fig.add_trace(
                    go.Heatmap(
                        x=times,
                        y=v_data if v_data.ndim == 1 else v_data[0],
                        z=y_data.T,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(
                            title=z_title, len=1.0 / num_plots, y=y_center, yanchor="middle", x=1.02
                        ),
                    ),
                    row=i + 1,
                    col=1,
                )

            else:
                # Line plot
                y_data = data.y

                # Configure a separate legend for this subplot
                legend_id = f"legend{i+1}" if i > 0 else "legend"
                layout_updates[legend_id] = dict(
                    y=y_center, yanchor="middle", x=1.02, xanchor="left", tracegroupgap=0
                )

                if y_data.ndim == 1:
                    name = format_latex(
                        legend_names[legend_idx] if legend_idx < len(legend_names) else var_name
                    )
                    legend_idx += 1
                    fig.add_trace(
                        go.Scatter(x=times, y=y_data, mode="lines", name=name, legend=legend_id),
                        row=i + 1,
                        col=1,
                    )
                else:
                    for col in range(y_data.shape[1]):
                        name = format_latex(
                            legend_names[legend_idx]
                            if legend_idx < len(legend_names)
                            else f"{var_name}_{col}"
                        )
                        legend_idx += 1
                        fig.add_trace(
                            go.Scatter(
                                x=times, y=y_data[:, col], mode="lines", name=name, legend=legend_id
                            ),
                            row=i + 1,
                            col=1,
                        )

        if y_type == "log":
            fig.update_yaxes(
                type=y_type, title_text=y_title, exponentformat="power", dtick=1, row=i + 1, col=1
            )
        else:
            fig.update_yaxes(type=y_type, title_text=y_title, row=i + 1, col=1)

    template = "plotly_dark" if dark_mode else "plotly_white"
    layout_updates.update(dict(height=250 * num_plots, template=template, showlegend=True))
    fig.update_layout(**layout_updates)

    return fig
