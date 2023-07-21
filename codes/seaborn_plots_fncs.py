import seaborn as sns
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

import SeabornFig2Grid as sfg


def kde_plots(
              df=None,
              x=None,
              y=None,
              x_label="",
              y_label="",
              data_type="",
              log_scale=True,
              x_log_scale=False,
              y_log_scale=False,
              xlim=[0, 2e1],
              ylim=[1e-1, 2e2],
              color="blue",
              marker_size=20,
              spearman=None,
              pearson=None,
              fig_save=True,
              hue_norm=mpl.colors.LogNorm(),
              height=8,
              ratio=8,
              space=0,
              alpha=0.7,
              bins=[20, 20],
              dark_mode=True,
              var_marker_size=False,
              ):

    pad = 5
    labelsize = 35
    ticklabelsize = 35
    clabelsize = 20
    ticklength = 15

    # Remove all occurances of delta_beta where delta_beta is less than 0
    if log_scale:
        df = df[df[x] > 0]
        df = df[df[y] > 0]

    axs1 = sns.JointGrid(x=x, y=y, data=df, xlim=xlim, ylim=ylim, height=height, ratio=ratio,
                         space=space, hue_norm=hue_norm)
    if y == "msh_msp_shear":
        axs1.plot_joint(sns.scatterplot, s=marker_size, alpha=alpha, color=color)
    else:
        axs1.plot_joint(sns.scatterplot, s=marker_size, alpha=alpha, color=color)

    # If var_marker_size is true, then make a concentric circle to show the size of the data points.
    # Make srue that the circles do not have a facecolor
    if var_marker_size:
        radii_max = np.sqrt(np.nanmax(marker_size) / 3)
        radii = np.linspace(1, radii_max, 4)
        radii = np.array([1, 3, 7, 12])
        radii_size = 3 * radii ** 2

        # Add scatter plot of circles to show the size of the data points at x=0 and y=0
        for i in range(len(radii)):
            axs1.ax_joint.scatter(x=[-7], y=[-8], s=radii_size[i], color='k', alpha=1,
                                  facecolor="none")
            # Add labels
            axs1.ax_joint.annotate(str(int(radii[i])), xy=(-6, -8.8 + i/1.2),
                                   textcoords="offset points", ha='center',
                                   va='center', fontsize=1.2*clabelsize, color='k', alpha=alpha)
        print(f"radii: {radii}")

    _ = sns.histplot(data=df, x=x, bins=bins[0], ax=axs1.ax_marg_x, legend=False, color=color,
                     alpha=alpha, kde=True, log_scale=x_log_scale, stat="frequency",
                     common_norm=True, common_bins=True, fill=True, linewidth=2, edgecolor=color,
                     line_kws={"linewidth": 5, "color": color})

    _ = sns.histplot(data=df, y=y, bins=bins[1], ax=axs1.ax_marg_y, legend=False, color=color,
                     alpha=alpha, kde=True, log_scale=y_log_scale, stat="frequency",
                     common_norm=True, common_bins=True, fill=True, linewidth=2, edgecolor=color,
                     line_kws={"linewidth": 5, "color": color})

    if y == "msh_msp_shear":
        shear_angle_theory = np.linspace(0, 180, 100)
        delta_beta_theory_half = np.tan(np.deg2rad(shear_angle_theory/2))
        delta_beta_theory_one = 2 * np.tan(np.deg2rad(shear_angle_theory/2))
        delta_beta_theory_two = 4 * np.tan(np.deg2rad(shear_angle_theory/2))

        axs1.fig.axes[0].plot(delta_beta_theory_half, shear_angle_theory, marker='.', ms=10, c="w",
                              ls=None, lw=0, label=r"$\lambda$ = 0.5")
        axs1.fig.axes[0].plot(delta_beta_theory_one, shear_angle_theory, marker='.', ms=10, c="b",
                              ls=None, lw=0, label=r"$\lambda$ = 1")
        axs1.fig.axes[0].plot(delta_beta_theory_two, shear_angle_theory, marker='.', ms=10, c="g",
                              ls=None, lw=0, label=r"$\lambda$ = 2")
        lgnd = axs1.fig.axes[0].legend(loc=2, fontsize=30, frameon=False)
        for handle in lgnd.legendHandles:
            handle.size = [1]

    if dark_mode:
        text_color = "white"
        face_color = "black"
        edge_color = "white"
    else:
        text_color = "black"
        face_color = "white"
        edge_color = "black"
    if y == "msh_msp_shear":
        spearman = None
    if spearman is not None:
        x_spearman = np.linspace(0, 25, 100)
        y_spearman = spearman * x_spearman + np.mean(df[y]) - spearman*np.mean(df[x])
        axs1.fig.axes[0].plot(x_spearman, y_spearman, c=color, ls="--", lw=5)
        axs1.fig.axes[0].text(0.02, 0.02, f"$\\rho_{{\\rm {'p'}}}$ = {spearman:.2f}",
                              transform=axs1.fig.axes[0].transAxes, va="bottom", ha="left",
                              bbox=dict(facecolor=face_color, alpha=1, edgecolor=edge_color,
                                        boxstyle='round,pad=0.2'), fontsize=ticklabelsize,
                              color=text_color)

    pos_joint_ax = axs1.ax_joint.get_position()
    pos_marg_x_ax = axs1.ax_marg_x.get_position()
    axs1.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width,
                                pos_joint_ax.height])
    axs1.fig.axes[-1].set_position([1, pos_joint_ax.y0, .07, pos_joint_ax.height])

    if data_type == "Shear" or data_type == "Reconnection-Energy":
        label_bottom = False
        x_label = ""
    else:
        label_bottom = True
        x_label = x_label

    axs1.fig.axes[0].tick_params(axis='both', which='major', direction='in',
                                 labelbottom=label_bottom,
                                 bottom=True, labeltop=False, top=True, labelleft=True, left=True,
                                 labelright=False, right=True, width=1.5, length=ticklength,
                                 labelsize=ticklabelsize, labelrotation=0, pad=pad)

    axs1.fig.axes[0].tick_params(axis='both', which='minor', direction='in', labelbottom=False,
                                 bottom=False, left=False, width=1.5, length=ticklength,
                                 labelsize=ticklabelsize, labelrotation=0)

    axs1.fig.axes[1].tick_params(axis='both', which='both', direction='in', labelbottom=False,
                                 bottom=False, labelleft=False, left=False, width=1.5,
                                 length=ticklength, labelsize=ticklabelsize, labelrotation=0)

    axs1.fig.axes[2].tick_params(axis='both', which='both', direction='in', labelbottom=False,
                                 bottom=False, labelleft=False, left=False, width=1.5,
                                 length=ticklength, labelsize=ticklabelsize, labelrotation=0)

    axs1.set_axis_labels(x_label, y_label, fontsize=labelsize, labelpad=-1)
    axs1.fig.axes[0].text(1, 0.02, f"{data_type}", transform=axs1.fig.axes[0].transAxes,
                          va="bottom", ha="right", bbox=dict(facecolor=face_color, alpha=1,
                          edgecolor=edge_color, boxstyle='round,pad=0.2'),
                          fontsize=ticklabelsize, color=text_color)

    # Set the tight layout
    axs1.fig.tight_layout()

    if (fig_save):
        fig_dir = "../figures/seaborn_plots/20230721/"
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fname = f"{fig_dir}/{x}_vs_{y}_{data_type}_dm_{dark_mode}_20230721.png"
        axs1.savefig(fname, format='png', dpi=400)
    plt.close('all')
    return axs1


def seaborn_subplots(
                     df_list=None,
                     keys=[],
                     figsize=(16, 16),
                     labels=[],
                     data_type=[],
                     color_list=[],
                     y_log_scale=False,
                     x_log_scale=False,
                     log_scale=True,
                     fig_name=None,
                     fig_format="png",
                     bins=None,
                     nbins=[20, 20],
                     x_lim=None,
                     y_lim=None,
                     dark_mode=False,
                     var_marker_size=True,
                     ):

    axs_list = []
    for i, df in enumerate(df_list):
        # Find the spearman and pearson correlation between key and "r_rc"
        spearman = df[keys[1]].corr(df["r_rc"], method="spearman")
        pearson = df[keys[1]].corr(df["r_rc"], method="pearson")

        if x_lim is None:
            x_lim = (df[keys[0]].min(), df[keys[0]].max())
        if x_log_scale and x_lim[0] <= 0:
            # Raise a warning saying that the minimum value was changed
            warnings.warn(f"\033[91m The minimum value of {keys[0]} was changed from "
                            f"{df[keys[0]].min():0.3f} to {x_lim[0]:0.3f} to avoid a log scale "
                            "error.\033[0m")
            # Set the minimum to minimum value greater than 0
            x_lim = (df[df[keys[0]] > 0][keys[0]].min(), x_lim[1])

        if y_lim is None:
            y_lim = [df[keys[1]].min(), df[keys[1]].max()]
        if y_log_scale and y_lim[0] <= 0:
            # Raise a warning saying that the minimum value was changed
            warnings.warn(f"\033[91m The minimum value of {keys[1]} was changed from "
                            f"{df[keys[1]].min():0.3f} to {y_lim[0]:0.3f} to avoid a log scale "
                            "error.\033[0m")

            # Set the minimum to minimum value greater than 0
            y_lim = (df[df[keys[1]] > 0][keys[1]].min(), y_lim[1])
        if bins is None and (x_log_scale or y_log_scale):
            if x_log_scale and y_log_scale:
                bins = [np.logspace(np.log10(x_lim[0]), np.log10(x_lim[1]), nbins[0]),
                        np.logspace(np.log10(y_lim[0]), np.log10(y_lim[1]), nbins[1])]
            elif x_log_scale and not y_log_scale:
                bins = [np.logspace(np.log10(x_lim[0]), np.log10(x_lim[1]), nbins[0]),
                        np.linspace(y_lim[0], y_lim[1], nbins[1])]
            elif not x_log_scale and y_log_scale:
                bins = [np.linspace(x_lim[0], x_lim[1], nbins[0]),
                        np.logspace(np.log10(y_lim[0]), np.log10(y_lim[1]), nbins[1])]
        elif bins is None and not (x_log_scale or y_log_scale):
            bins = [np.linspace(x_lim[0], x_lim[1], nbins[0]),
                    np.linspace(y_lim[0], y_lim[1], nbins[1])]

        if var_marker_size:
            marker_size = 3 * df.r_rc.values**2
            spearman = None
        else:
            marker_size = 100
        axs = kde_plots(df=df, x=keys[0], y=keys[1], x_label=labels[0],
                        y_label=labels[1], data_type=data_type[i], log_scale=log_scale,
                        x_log_scale=x_log_scale, y_log_scale=y_log_scale, marker_size=marker_size,
                        xlim=x_lim, ylim=y_lim, color=color_list[i], spearman=spearman,
                        pearson=pearson, fig_save=True, bins=bins, dark_mode=dark_mode,
                        var_marker_size=var_marker_size)
        axs_list.append(axs)

    print(f"The figure size is {figsize[0]}, {figsize[1]}")
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    gs = gridspec.GridSpec(2, 2)

    _ = sfg.SeabornFig2Grid(axs_list[0], fig, gs[0])
    _ = sfg.SeabornFig2Grid(axs_list[1], fig, gs[1])
    _ = sfg.SeabornFig2Grid(axs_list[2], fig, gs[2])
    _ = sfg.SeabornFig2Grid(axs_list[3], fig, gs[3])

    gs.tight_layout(fig)
    gs.update(top=1, bottom=0.05, left=0.085, right=1, hspace=0.01, wspace=0.22)
    if fig_name is None:
        fig_name = f"../figures/seaborn_plots/20230721/{keys[0]}_vs_{keys[1]}_dm_{dark_mode}" +\
                   f"_20230721.{fig_format}"
    else:
        fig_name = f"../figures/seaborn_plots/20230721/{fig_name}_{dark_mode}_20230721.{fig_format}"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight', pad_inches=0.25, format=fig_format)
    print(f"Saved figure to {fig_name} for {keys[0]} vs {keys[1]}")

    return axs_list
