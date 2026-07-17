"""
Interactive Statistics Plots Module

Generates interactive Plotly figures for the statistics mode of the GUI.
This is the interactive counterpart to app_stats_plots.py (which generates
static matplotlib figures). All plots are rendered as Plotly figures that
support zoom, pan, hover tooltips, and data selection.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rxn_location import rc_stats_fncs as rcsf


def generate_interactive_plots(
    csv_file_path,
    dark_mode=False,
    selected_plots=None,
    x_var="b_imf_z",
    y_var="r_rc",
):
    """
    Generate interactive Plotly figures from a statistics CSV file.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file containing statistics data.
    dark_mode : bool
        If True, use a dark background template.
    selected_plots : list of str
        Which plot types to generate. Options: "Histograms", "KDE Plots",
        "2D Histograms", "Scatter Plots", "MMS Location Scatter Plot".
    x_var : str
        Column name for the x-axis variable.
    y_var : str
        Column name for the y-axis variable.

    Returns
    -------
    figures : dict
        Mapping of plot title to Plotly Figure object.
    err : str or None
        Error message if something went wrong, else None.
    """
    if selected_plots is None:
        selected_plots = ["Scatter Plots"]

    figures = {}
    template = "plotly_dark" if dark_mode else "plotly_white"

    # Label mappings
    var_labels = {
        "b_imf_z": "IMF Bz [nT]",
        "b_imf_x": "IMF Bx [nT]",
        "b_imf_y": "IMF By [nT]",
        "imf_clock_angle": "IMF Clock Angle [°]",
        "cone_angle": "Cone Angle [°]",
        "p_dyn": "Dynamic Pressure [nPa]",
        "msh_msp_shear": "Shear Angle [°]",
        "r_rc": "Reconnection Distance [Rₑ]",
        "delta_beta": "Δβ",
    }

    x_label = var_labels.get(x_var, x_var.replace("_", " ").title())
    y_label = var_labels.get(y_var, y_var.replace("_", " ").title())

    # Read and prepare data
    try:
        df = pd.read_csv(csv_file_path, index_col=False)
        df = rcsf.convert_wide_to_long(df)
    except Exception as e:
        return figures, f"Error reading CSV: {e}"

    # Split by model
    model_names = ["Shear", "Reconnection Energy", "Exhaust Velocity", "Bisection"]
    model_filters = [
        ["shear", "Shear"],
        ["rx_en", "Reconnection Energy"],
        ["va_cs", "Exhaust Velocity"],
        ["bisection", "Bisection Field"],
    ]
    color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    df_list = []
    for filters in model_filters:
        df_model = df[df.method_used.isin(filters)].copy()
        # Compute derived columns if possible
        if len(df_model) > 0:
            if "imf_clock_angle" in df_model.columns:
                df_model["imf_clock_angle"] = df_model["imf_clock_angle"].apply(
                    lambda x: 360 - x if x > 180 else x
                )
            if all(c in df_model.columns for c in ["b_imf_x", "b_imf_y", "b_imf_z"]):
                mag = np.sqrt(
                    df_model.b_imf_x**2 + df_model.b_imf_y**2 + df_model.b_imf_z**2
                )
                df_model["cone_angle"] = np.arccos(df_model.b_imf_x / mag) * 180 / np.pi
                df_model["bb"] = df_model.b_imf_y / mag
        df_list.append(df_model)

    # ── Histograms ──────────────────────────────────────────────────────
    if "Histograms" in selected_plots:
        try:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=model_names,
                vertical_spacing=0.12,
                horizontal_spacing=0.08,
            )
            for i, (df_n, name) in enumerate(zip(df_list, model_names)):
                row, col = i // 2 + 1, i % 2 + 1
                if len(df_n) > 0 and "r_rc" in df_n.columns:
                    vals = df_n["r_rc"].dropna()
                    if len(vals) > 0:
                        fig.add_trace(
                            go.Histogram(
                                x=vals,
                                name=name,
                                marker_color=color_list[i],
                                opacity=0.8,
                                showlegend=False,
                            ),
                            row=row,
                            col=col,
                        )
                        fig.update_xaxes(
                            title_text="Reconnection Distance [Rₑ]" if row == 2 else "",
                            row=row,
                            col=col,
                        )
                        fig.update_yaxes(
                            title_text="Count" if col == 1 else "",
                            row=row,
                            col=col,
                        )

            fig.update_layout(
                template=template,
                height=700,
                title_text="Reconnection Distance Histograms",
                title_x=0.5,
            )
            figures["Histograms"] = fig
        except Exception as e:
            print(f"Interactive Histogram error: {e}")

    # ── 2D Histograms (Heatmaps) ────────────────────────────────────────
    if "2D Histograms" in selected_plots:
        try:
            colorscales = ["Viridis", "Cividis", "Plasma", "Magma"]
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=model_names,
                vertical_spacing=0.12,
                horizontal_spacing=0.10,
            )
            for i, (df_n, name) in enumerate(zip(df_list, model_names)):
                row, col = i // 2 + 1, i % 2 + 1
                if len(df_n) > 0 and x_var in df_n.columns and y_var in df_n.columns:
                    fig.add_trace(
                        go.Histogram2d(
                            x=df_n[x_var],
                            y=df_n[y_var],
                            colorscale=colorscales[i],
                            nbinsx=25,
                            nbinsy=25,
                            showscale=True,
                            name=name,
                        ),
                        row=row,
                        col=col,
                    )
                fig.update_xaxes(
                    title_text=x_label if row == 2 else "",
                    row=row,
                    col=col,
                )
                fig.update_yaxes(
                    title_text=y_label if col == 1 else "",
                    row=row,
                    col=col,
                )

            fig.update_layout(
                template=template,
                height=700,
                title_text=f"2D Histograms: {x_label} vs {y_label}",
                title_x=0.5,
            )
            figures["2D Histograms"] = fig
        except Exception as e:
            print(f"Interactive 2D Histogram error: {e}")

    # ── Scatter Plots ───────────────────────────────────────────────────
    if "Scatter Plots" in selected_plots:
        try:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=model_names,
                vertical_spacing=0.12,
                horizontal_spacing=0.08,
            )

            for i, (df_n, name) in enumerate(zip(df_list, model_names)):
                row, col = i // 2 + 1, i % 2 + 1
                if len(df_n) > 0 and x_var in df_n.columns and y_var in df_n.columns:
                    # Build hover text with key metadata
                    hover_parts = [
                        f"<b>{name}</b>",
                        f"{x_label}: %{{x:.3f}}",
                        f"{y_label}: %{{y:.3f}}",
                    ]
                    # Add jet_time if available
                    custom_data = []
                    extra_hover = []
                    if "jet_time" in df_n.columns:
                        custom_data.append(df_n["jet_time"].astype(str).values)
                        extra_hover.append("Jet Time: %{customdata[0]}")
                    if "Date" in df_n.columns:
                        custom_data.append(df_n["Date"].astype(str).values)
                        extra_hover.append(
                            "Date: %{customdata[" + str(len(custom_data) - 1) + "]}"
                        )

                    hovertemplate = (
                        "<br>".join(hover_parts + extra_hover) + "<extra></extra>"
                    )

                    scatter_kwargs = dict(
                        x=df_n[x_var],
                        y=df_n[y_var],
                        mode="markers",
                        name=name,
                        marker=dict(
                            color=color_list[i],
                            size=7,
                            opacity=0.7,
                            line=dict(
                                width=0.5, color="white" if dark_mode else "black"
                            ),
                        ),
                        hovertemplate=hovertemplate,
                        showlegend=False,
                    )
                    if custom_data:
                        scatter_kwargs["customdata"] = np.column_stack(custom_data)

                    fig.add_trace(go.Scatter(**scatter_kwargs), row=row, col=col)

                fig.update_xaxes(
                    title_text=x_label if row == 2 else "",
                    row=row,
                    col=col,
                )
                fig.update_yaxes(
                    title_text=y_label if col == 1 else "",
                    row=row,
                    col=col,
                )

            fig.update_layout(
                template=template,
                height=700,
                title_text=f"Scatter: {x_label} vs {y_label}",
                title_x=0.5,
            )
            figures["Scatter Plots"] = fig
        except Exception as e:
            print(f"Interactive Scatter Plot error: {e}")

    # ── MMS Location Scatter Plot ───────────────────────────────────────
    if "MMS Location Scatter Plot" in selected_plots:
        try:
            pos_cols_xy = ["spc_pos_x", "spc_pos_y"]
            pos_cols_yz = ["spc_pos_y", "spc_pos_z"]
            # Also check for master-list column names
            alt_pos_cols_xy = ["x_gsm", "y_gsm"]
            alt_pos_cols_yz = ["y_gsm", "z_gsm"]

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["Spacecraft Y vs X", "Spacecraft Z vs Y"],
                horizontal_spacing=0.10,
            )
            for i, (df_n, name) in enumerate(zip(df_list, model_names)):
                if len(df_n) == 0:
                    continue

                # Determine which column names exist
                if all(c in df_n.columns for c in pos_cols_xy):
                    x_col, y_col = pos_cols_xy
                    y2_col, z_col = pos_cols_yz
                elif all(c in df_n.columns for c in alt_pos_cols_xy):
                    x_col, y_col = alt_pos_cols_xy
                    y2_col, z_col = alt_pos_cols_yz
                else:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=df_n[x_col],
                        y=df_n[y_col],
                        mode="markers",
                        name=name,
                        marker=dict(color=color_list[i], size=6, opacity=0.7),
                        legendgroup=name,
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_n[y2_col],
                        y=df_n[z_col],
                        mode="markers",
                        name=name,
                        marker=dict(color=color_list[i], size=6, opacity=0.7),
                        legendgroup=name,
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

            fig.update_xaxes(title_text="MMS Position X [Rₑ]", row=1, col=1)
            fig.update_yaxes(title_text="MMS Position Y [Rₑ]", row=1, col=1)
            fig.update_xaxes(title_text="MMS Position Y [Rₑ]", row=1, col=2)
            fig.update_yaxes(title_text="MMS Position Z [Rₑ]", row=1, col=2)

            fig.update_layout(
                template=template,
                height=500,
                title_text="MMS Spacecraft Positions",
                title_x=0.5,
            )
            figures["MMS Location Scatter Plot"] = fig
        except Exception as e:
            print(f"Interactive MMS Location Plot error: {e}")

    return figures, None
