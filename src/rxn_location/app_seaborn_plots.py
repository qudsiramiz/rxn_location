"""
Seaborn-based statistical plots for the Streamlit GUI.

Adapted from seaborn_plots_fncs.py and SeabornFig2Grid.py for in-app rendering.
These functions are designed to return matplotlib Figure objects for use with
st.pyplot(), without saving to disk.
"""

import seaborn as sns
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import warnings

# ============================================================================
# SeabornFig2Grid — inline copy to avoid modifying the original
# ============================================================================


class _SeabornFig2Grid:
    """Move a seaborn JointGrid into a subplot of an existing matplotlib figure."""

    def __init__(self, seaborngrid, fig, subplot_spec):
        """
        Initializes the Streamlit-compatible Seaborn Figure wrapper.
        """
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or isinstance(
            self.sg, sns.axisgrid.PairGrid
        ):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """
        Overrides the seaborn Grid plotting logic to redirect axes to the Streamlit layout.
        """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n, m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i, j], self.subgrid[i, j])

    def _movejointgrid(self):
        """
        Overrides the seaborn JointGrid plotting logic to redirect axes to the Streamlit layout.
        """
        h = self.sg.ax_joint.get_position().height
        h2 = self.sg.ax_marg_x.get_position().height
        r = int(np.round(h / h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(
            r + 1, r + 1, subplot_spec=self.subplot
        )
        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        """
        Overrides individual seaborn axes rendering to capture the output for Streamlit.
        """
        ax.remove()
        ax.figure = self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        """
        Finalizes the figure state and pushes it to Streamlit's rendering queue.
        """
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        """
        Handles responsive resizing of the seaborn figures in the Streamlit web layout.
        """
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


# ============================================================================
# Core plotting functions
# ============================================================================


def _kde_plot_panel(
    df,
    x,
    y,
    x_label="",
    y_label="",
    data_type="",
    log_scale=False,
    x_log_scale=False,
    y_log_scale=False,
    xlim=(0, 20),
    ylim=(-1, 2),
    color="blue",
    marker_size=20,
    spearman=None,
    bins=(20, 20),
    dark_mode=False,
    marker_size_var="None",
    alpha=0.7,
    height=8,
    ratio=8,
    space=0,
):
    """
    Build a single seaborn JointGrid panel (scatter + marginal histograms).

    Returns the JointGrid object (with its own internal figure) so it can be
    composited into a 2×2 grid by the caller.
    """

    pad = 3
    labelsize = 18
    ticklabelsize = 16
    clabelsize = 14
    ticklength = 8

    if log_scale:
        df = df[df[x] > 0]
        df = df[df[y] > 0]

    jg = sns.JointGrid(
        x=x,
        y=y,
        data=df,
        xlim=xlim,
        ylim=ylim,
        height=height,
        ratio=ratio,
        space=space,
    )
    jg.ax_joint.scatter(df[x], df[y], s=marker_size, alpha=alpha, color=color)

    # Variable marker size legend (concentric circles)
    if marker_size_var == "r_rc":
        radii = np.array([1, 3, 7, 12])
        radii_size = 3 * radii**2
        # Add scatter plot of circles to show the size of the data points at x=0 and y=0

        for i in range(len(radii)):
            jg.ax_joint.scatter(
                x=[xlim[0] + 1],
                y=[ylim[0] + 0.5],
                s=radii_size[i],
                color="w",
                alpha=1,
                facecolor="none",
            )
            jg.ax_joint.annotate(
                str(int(radii[i])),
                xy=(xlim[0] + 2, ylim[0] + 0.7 + i / 1.2),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=1.2 * clabelsize,
                color="k",
                alpha=alpha,
            )

    # Marginal histograms
    sns.histplot(
        data=df,
        x=x,
        bins=bins[0],
        ax=jg.ax_marg_x,
        legend=False,
        color=color,
        alpha=alpha,
        kde=True,
        log_scale=x_log_scale,
        stat="frequency",
        common_norm=True,
        common_bins=True,
        fill=True,
        linewidth=2,
        edgecolor=color,
        line_kws={"linewidth": 5, "color": color},
    )
    sns.histplot(
        data=df,
        y=y,
        bins=bins[1],
        ax=jg.ax_marg_y,
        legend=False,
        color=color,
        alpha=alpha,
        kde=True,
        log_scale=y_log_scale,
        stat="frequency",
        common_norm=True,
        common_bins=True,
        fill=True,
        linewidth=2,
        edgecolor=color,
        line_kws={"linewidth": 5, "color": color},
    )

    # Styling
    if dark_mode:
        text_color, face_color, edge_color = "white", "black", "white"
    else:
        text_color, face_color, edge_color = "black", "white", "black"

    # Spearman correlation line
    if spearman is not None:
        x_line = np.linspace(xlim[0], xlim[1], 100)
        y_line = spearman * x_line + np.mean(df[y]) - spearman * np.mean(df[x])
        jg.fig.axes[0].plot(x_line, y_line, c=color, ls="--", lw=5)
        jg.fig.axes[0].text(
            0.02,
            0.02,
            f"$\\rho_{{\\rm p}}$ = {spearman:.2f}",
            transform=jg.fig.axes[0].transAxes,
            va="bottom",
            ha="left",
            bbox=dict(
                facecolor=face_color,
                alpha=1,
                edgecolor=edge_color,
                boxstyle="round,pad=0.2",
            ),
            fontsize=ticklabelsize,
            color=text_color,
        )

    # Axis alignment
    pos_joint = jg.ax_joint.get_position()
    pos_marg_x = jg.ax_marg_x.get_position()
    jg.ax_joint.set_position(
        [pos_joint.x0, pos_joint.y0, pos_marg_x.width, pos_joint.height]
    )
    jg.fig.axes[-1].set_position([1, pos_joint.y0, 0.07, pos_joint.height])

    # Conditionally hide x-label for top-row panels
    if data_type in ("Shear", "Reconnection-Energy"):
        label_bottom = False
        x_label_final = ""
    else:
        label_bottom = True
        x_label_final = x_label

    jg.fig.axes[0].tick_params(
        axis="both",
        which="major",
        direction="in",
        labelbottom=label_bottom,
        bottom=True,
        labeltop=False,
        top=True,
        labelleft=True,
        left=True,
        labelright=False,
        right=True,
        width=1.5,
        length=ticklength,
        labelsize=ticklabelsize,
        labelrotation=0,
        pad=pad,
    )
    jg.fig.axes[0].tick_params(
        axis="both",
        which="minor",
        direction="in",
        labelbottom=False,
        bottom=False,
        left=False,
        width=1.5,
        length=ticklength,
        labelsize=ticklabelsize,
        labelrotation=0,
    )
    for ax_idx in (1, 2):
        if ax_idx < len(jg.fig.axes):
            jg.fig.axes[ax_idx].tick_params(
                axis="both",
                which="both",
                direction="in",
                labelbottom=False,
                bottom=False,
                labelleft=False,
                left=False,
                width=1.5,
                length=ticklength,
                labelsize=ticklabelsize,
                labelrotation=0,
            )

    jg.set_axis_labels(x_label_final, y_label, fontsize=labelsize, labelpad=-1)
    jg.fig.axes[0].text(
        1,
        0.02,
        f"{data_type}",
        transform=jg.fig.axes[0].transAxes,
        va="bottom",
        ha="right",
        bbox=dict(
            facecolor=face_color,
            alpha=1,
            edgecolor=edge_color,
            boxstyle="round,pad=0.2",
        ),
        fontsize=ticklabelsize,
        color=text_color,
    )
    jg.fig.tight_layout()

    return jg


def generate_seaborn_jointplots(
    df_full,
    x_key="r_rc",
    y_key="b_imf_z",
    x_label="Reconnection Distance [R_E]",
    y_label="IMF Bz [nT]",
    x_lim=None,
    y_lim=None,
    dark_mode=False,
    marker_size_var="None",
    nbins=(40, 40),
    figsize=(16, 16),
):
    """
    Generate a 2×2 grid of seaborn joint-plots (one per reconnection model)
    from a statistics DataFrame.

    Parameters
    ----------
    df_full : pandas.DataFrame
        Full reconnection statistics dataframe containing a ``method_used``
        column used to split data into the four models.
    x_key, y_key : str
        Column names for the x- and y-axis variables.
    x_label, y_label : str
        Human-readable axis labels.
    x_lim, y_lim : tuple of float, optional
        Axis limits.  Auto-calculated when *None*.
    dark_mode : bool
        Apply dark background styling.
    marker_size_var : str
        Scale marker size by the specified column name.
    nbins : tuple of int
        Number of bins for (x, y) marginal histograms.
    figsize : tuple of int
        Overall figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The composited 2×2 figure ready for ``st.pyplot(fig)``.
    """

    # --- Set style ---
    if dark_mode:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    font = {"family": "serif", "weight": "normal", "size": 10}
    plt.rc("font", **font)
    # Do NOT enable usetex — Streamlit servers rarely have LaTeX installed
    plt.rc("text", usetex=False)

    # --- Split by model ---
    model_map = {
        "shear": "Shear",
        "rx_en": "Reconnection-Energy",
        "va_cs": "Exhaust-Velocity",
        "bisection": "Bisection",
    }
    # Normalise the method_used values coming from the CSV
    method_col = df_full["method_used"].str.strip().str.lower()

    # Build sub-dataframes in canonical order
    canonical_order = ["shear", "rx_en", "va_cs", "bisection"]
    # Create a mapping from possible CSV values → canonical key
    alias_map = {
        "shear": "shear",
        "reconnection energy": "rx_en",
        "reconnection-energy": "rx_en",
        "rx_en": "rx_en",
        "exhaust velocity": "va_cs",
        "exhaust-velocity": "va_cs",
        "va_cs": "va_cs",
        "bisection": "bisection",
        "bisection field": "bisection",
    }

    df_dict = {}
    for canon in canonical_order:
        matching_aliases = [k for k, v in alias_map.items() if v == canon]
        mask = method_col.isin(matching_aliases)
        df_dict[canon] = df_full[mask].copy()

    df_list = [df_dict[c] for c in canonical_order]
    data_type = [model_map[c] for c in canonical_order]
    color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # --- Compute derived columns if needed ---
    # Cone angle
    if "cone_angle" not in df_full.columns:
        # Use shear df as reference for b_imf components
        ref = df_list[0] if len(df_list[0]) > 0 else df_full
        if all(c in ref.columns for c in ("b_imf_x", "b_imf_y", "b_imf_z")):
            bmag = np.sqrt(
                ref["b_imf_x"] ** 2 + ref["b_imf_y"] ** 2 + ref["b_imf_z"] ** 2
            )
            cone = np.arccos(ref["b_imf_x"] / bmag) * 180 / np.pi
            for dfn in df_list:
                if len(dfn) > 0:
                    dfn["cone_angle"] = cone.reindex(dfn.index).values

    # By/|B| ratio
    if "bb" not in df_full.columns:
        ref = df_list[0] if len(df_list[0]) > 0 else df_full
        if all(c in ref.columns for c in ("b_imf_x", "b_imf_y", "b_imf_z")):
            bmag = np.sqrt(
                ref["b_imf_x"] ** 2 + ref["b_imf_y"] ** 2 + ref["b_imf_z"] ** 2
            )
            bb_ratio = ref["b_imf_y"] / bmag
            for dfn in df_list:
                if len(dfn) > 0:
                    dfn["bb"] = bb_ratio.reindex(dfn.index).values

    # Helper for model-specific Rc keys
    model_map_rc = {
        "shear": "r_rc_Shear",
        "bisection": "r_rc_Bisection Field",
        "rx_en": "r_rc_Reconnection Energy",
        "va_cs": "r_rc_Exhaust Velocity",
    }

    def get_actual_key(base_key, canon_model, test_df):
        """
        Helper to map user-friendly label names back to their raw master jet list keys.
        """
        if base_key.startswith("r_rc"):
            target = model_map_rc.get(canon_model)
            if target:
                suffix = base_key[4:]
                if suffix and not suffix.startswith("_"):
                    suffix = f"_{suffix}"
                target_with_suffix = f"{target}{suffix}"
                if f"data_{target_with_suffix}" in test_df.columns:
                    return f"data_{target_with_suffix}"
                if target_with_suffix in test_df.columns:
                    return target_with_suffix
        return base_key

    # --- Auto-calculate limits ---
    if x_lim is None:
        all_x = []
        for c, df in df_dict.items():
            k = get_actual_key(x_key, c, df_full)
            if len(df) > 0 and k in df.columns:
                all_x.append(df[k].dropna().values)
        if all_x:
            combined = np.concatenate(all_x)
            x_lim = (float(np.nanmin(combined)), float(np.nanmax(combined)))
        else:
            x_lim = (0, 1)
        if x_lim[0] == x_lim[1]:
            x_lim = (x_lim[0] - 1, x_lim[1] + 1)

    if y_lim is None:
        all_y = []
        for c, df in df_dict.items():
            k = get_actual_key(y_key, c, df_full)
            if len(df) > 0 and k in df.columns:
                all_y.append(df[k].dropna().values)
        if all_y:
            combined = np.concatenate(all_y)
            y_lim = (float(np.nanmin(combined)), float(np.nanmax(combined)))
        else:
            y_lim = (0, 1)
        if y_lim[0] == y_lim[1]:
            y_lim = (y_lim[0] - 1, y_lim[1] + 1)

    # --- Build individual JointGrid panels ---
    panels = []
    bins = [
        np.linspace(x_lim[0], x_lim[1], nbins[0]),
        np.linspace(y_lim[0], y_lim[1], nbins[1]),
    ]

    for i, (canon_model, df) in enumerate(df_dict.items()):
        act_x = get_actual_key(x_key, canon_model, df_full)
        act_y = get_actual_key(y_key, canon_model, df_full)
        act_marker = get_actual_key(marker_size_var, canon_model, df_full)

        if len(df) == 0 or act_x not in df.columns or act_y not in df.columns:
            panels.append(None)
            continue

        spearman = (
            df[act_y].corr(df[act_x], method="spearman")
            if act_x in df.columns and marker_size_var == "None"
            else None
        )

        msz = 100
        if marker_size_var != "None" and act_marker in df.columns:
            if marker_size_var.startswith("r_rc"):
                msz = 3 * df[act_marker].values ** 2
            else:
                vals = df[act_marker].values
                v_min = np.nanmin(vals)
                v_max = np.nanmax(vals)
                if v_max > v_min:
                    norm_vals = (vals - v_min) / (v_max - v_min)
                    msz = 20 + norm_vals * 280
                else:
                    msz = 100

        jg = _kde_plot_panel(
            df=df,
            x=act_x,
            y=act_y,
            x_label=x_label,
            y_label=y_label,
            data_type=data_type[i],
            xlim=x_lim,
            ylim=y_lim,
            color=color_list[i],
            marker_size=msz,
            spearman=spearman,
            bins=bins,
            dark_mode=dark_mode,
            marker_size_var=marker_size_var,
        )
        panels.append(jg)

    # --- Composite into a single 2×2 figure ---
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2)

    for idx, panel in enumerate(panels):
        if panel is not None:
            _SeabornFig2Grid(panel, fig, gs[idx])

    gs.tight_layout(fig, rect=[0.02, 0.02, 0.98, 0.98])
    gs.update(top=0.95, bottom=0.08, left=0.1, right=0.95, hspace=0.05, wspace=0.25)

    # Close any stray panel figures to prevent memory leaks
    plt.close("all")
    # Re-register our composited figure so it stays alive
    # (plt.close("all") would have closed it too)
    # Instead, we just return it — the caller will render with st.pyplot(fig)

    return fig
