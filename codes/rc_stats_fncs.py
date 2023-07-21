import importlib
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from matplotlib.pyplot import MaxNLocator
import rx_model_funcs as rmf

importlib.reload(rmf)

# Set the font size for the axes
label_size = 20  # fontsize for x and y labels
t_label_size = 18  # fontsize for tick label
c_label_size = 18  # fontsize for colorbar label
ct_tick_size = 14  # fontsize for colorbar tick labels
l_label_size = 14  # fontsize for legend label

tick_len = 12  # length of the tick lines
mtick_len = 7  # length of the minor tick lines
tick_width = 1  # tick width in points
mtick_width = 0.7  # minor tick width in points

label_pad = 5  # padding between label and axis


def plot_hist(file_name, fig_size=(6, 6), dark_mode=True, bins=8, fig_folder="../figures",
              fig_name="new", fig_format="pdf", histtype="step", linewidth=1, cut_type="jet",
              r_lim=[0, 15], density=False):

    df = pd.read_csv(file_name, index_col=False)

    # Set date_from as index
    df = df.set_index("date_from")

    df_shear = df[df.method_used == "shear"].copy()
    df_rx_en = df[df.method_used == "rx_en"].copy()
    df_va_cs = df[df.method_used == "va_cs"].copy()
    df_bisec = df[df.method_used == "bisection"].copy()

    cone_angle_shear = np.arccos(df_shear.b_imf_x / np.sqrt(
                       df_shear.b_imf_x**2 +df_shear.b_imf_y**2+ df_shear.b_imf_z**2)) * 180 / np.pi
    cone_angle_rx_en = np.arccos(df_rx_en.b_imf_x / np.sqrt(
                          df_rx_en.b_imf_x**2 +df_rx_en.b_imf_y**2+ df_rx_en.b_imf_z**2)) * 180 / np.pi
    cone_angle_va_cs = np.arccos(df_va_cs.b_imf_x / np.sqrt(
                            df_va_cs.b_imf_x**2 +df_va_cs.b_imf_y**2+ df_va_cs.b_imf_z**2)) * 180 / np.pi
    cone_angle_bisec = np.arccos(df_bisec.b_imf_x / np.sqrt(
                            df_bisec.b_imf_x**2 +df_bisec.b_imf_y**2+ df_bisec.b_imf_z**2)) * 180 / np.pi

    df_shear["cone_angle"] = cone_angle_shear
    df_rx_en["cone_angle"] = cone_angle_rx_en
    df_va_cs["cone_angle"] = cone_angle_va_cs
    df_bisec["cone_angle"] = cone_angle_bisec

    if cut_type == "bz_neg":
        df_shear = df_shear[df_shear["b_imf_z"] < 0]
        df_rx_en = df_rx_en[df_rx_en["b_imf_z"] < 0]
        df_va_cs = df_va_cs[df_va_cs["b_imf_z"] < 0]
        df_bisec = df_bisec[df_bisec["b_imf_z"] < 0]

    if cut_type == "bz_pos":
        df_shear = df_shear[df_shear["b_imf_z"] > 0]
        df_rx_en = df_rx_en[df_rx_en["b_imf_z"] > 0]
        df_va_cs = df_va_cs[df_va_cs["b_imf_z"] > 0]
        df_bisec = df_bisec[df_bisec["b_imf_z"] > 0]
    
    if cut_type == "cone_angle":
        # Select all data where cone angle is greater than 36.87 degrees
        df_shear = df_shear[(df_shear.cone_angle <= 36.87)]
        df_rx_en = df_rx_en[(df_rx_en.cone_angle <= 36.87)]
        df_va_cs = df_va_cs[(df_va_cs.cone_angle <= 36.87)]
        df_bisec = df_bisec[(df_bisec.cone_angle <= 36.87)]

    if cut_type == "cone_and_bz_neg":
        df_shear = df_shear[(df_shear.cone_angle >= 36.87) & (df_shear["b_imf_z"] < 0) & (df_shear.cone_angle <= 150)]
        df_rx_en = df_rx_en[(df_rx_en.cone_angle >= 36.87) & (df_rx_en["b_imf_z"] < 0) & (df_rx_en.cone_angle <= 150)]
        df_va_cs = df_va_cs[(df_va_cs.cone_angle >= 36.87) & (df_va_cs["b_imf_z"] < 0) & (df_va_cs.cone_angle <= 150)]
        df_bisec = df_bisec[(df_bisec.cone_angle >= 36.87) & (df_bisec["b_imf_z"] < 0) & (df_bisec.cone_angle <= 150)]

    if dark_mode:
        plt.style.use('dark_background')
        # tick_color = 'w'  # color of the tick lines
        mtick_color = 'w'  # color of the minor tick lines
        label_color = 'w'  # color of the tick labels
        # clabel_color = 'w'  # color of the colorbar label
    else:
        plt.style.use('default')
        # tick_color = 'k'  # color of the tick lines
        mtick_color = 'k'  # color of the minor tick lines
        label_color = 'k'  # color of the tick labels
        # clabel_color = 'k'  # color of the colorbar label

    # Set the fontstyle to Times New Roman
    font = {'family': 'serif', 'weight': 'normal', 'size': 10}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)

    plt.close("all")

    r_lim_val = 12
    shear_r_rc_mean = df_shear[df_shear.r_rc < r_lim_val]["r_rc"].mean()
    rx_en_r_rc_mean = df_rx_en[df_rx_en.r_rc < r_lim_val]["r_rc"].mean()
    va_cs_r_rc_mean = df_va_cs[df_va_cs.r_rc < r_lim_val]["r_rc"].mean()
    bisec_r_rc_mean = df_bisec[df_bisec.r_rc < r_lim_val]["r_rc"].mean()
    if density==True:
        y_label = "Frequency"
    else:
        y_label = "Counts"
    fig = plt.figure(num=None, figsize=fig_size, dpi=200, facecolor='k', edgecolor='w')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0., hspace=0.)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])

    # Plot the histogram of the shear data
    axs1 = plt.subplot(gs[0, 0])
    axs1.hist(df_shear.r_rc, bins=bins, range=(0, 15), color='#1f77b4', alpha=0.5, density=density,
              histtype=histtype, linewidth=linewidth)
    # Plot the median of the shear data and add atext to the line
    axs1.axvline(shear_r_rc_mean, color='#1f77b4', linestyle='--', linewidth=2)
    axs1.text(shear_r_rc_mean+0.2, axs1.get_ylim()[1]*0.2,
              "$R_{{\\rm{{rc}}}}$ = {:.2f}$R_{{\\rm E}}$".format(shear_r_rc_mean),
              fontsize=1.1*t_label_size, color=label_color)
    axs1.text(0.05, 0.9, "(a)", fontsize=1.1*t_label_size, color=label_color,
              transform=axs1.transAxes)
    axs1.set_xlim(r_lim[0], r_lim[1])
    axs1.set_xscale('linear')
    # axs1.set_xlabel(r'$r_{rc}$', fontsize=label_size, color=label_color, labelpad=label_pad)
    axs1.set_ylabel(y_label, fontsize=label_size, color=label_color, labelpad=label_pad)

    # Plot the histogram of the rx_en data
    axs2 = plt.subplot(gs[0, 1])
    axs2.hist(df_rx_en.r_rc, bins=bins, range=(0, 15), color='#ff7f0e', alpha=0.5, density=density,
              histtype=histtype, linewidth=linewidth)
    # Plot the median of the rx_en data and add atext to the line
    axs2.axvline(rx_en_r_rc_mean, color='#ff7f0e', linestyle='--', linewidth=2)
    axs2.text(rx_en_r_rc_mean+0.2, axs2.get_ylim()[1]*0.2,
              "$R_{{\\rm{{rc}}}}$ = {:.2f}$R_{{\\rm E}}$".format(rx_en_r_rc_mean),
              fontsize=1.1*t_label_size, color=label_color)
    axs2.text(0.05, 0.9, "(b)", fontsize=1.1*t_label_size, color=label_color,
              transform=axs2.transAxes)
    axs2.set_xlim(r_lim[0], r_lim[1])
    axs2.set_xscale('linear')
    # axs2.set_xlabel(r'$r_{rc}$', fontsize=label_size, color=label_color, labelpad=label_pad)
    axs2.set_ylabel(y_label, fontsize=label_size, color=label_color, labelpad=label_pad)
    axs2.yaxis.set_label_position("right")

    # Plot the histogram of the va_cs data
    axs3 = plt.subplot(gs[1, 0])
    axs3.hist(df_va_cs.r_rc, bins=bins, range=(0, 15), color='#2ca02c', alpha=0.5, density=density,
              histtype=histtype, linewidth=linewidth)
    # Plot the median of the va_cs data and add atext to the line
    axs3.axvline(va_cs_r_rc_mean, color='#2ca02c', linestyle='--', linewidth=2)
    axs3.text(va_cs_r_rc_mean+0.2, axs3.get_ylim()[1]*0.2,
              "$R_{{\\rm{{rc}}}}$ = {:.2f}$R_{{\\rm E}}$".format(va_cs_r_rc_mean),
              fontsize=1.1*t_label_size, color=label_color)
    axs3.text(0.05, 0.9, "(c)", fontsize=1.1*t_label_size, color=label_color,
              transform=axs3.transAxes)
    axs3.set_xlim(r_lim[0], r_lim[1])
    axs3.set_xscale('linear')
    axs3.set_xlabel(r'$R_{\rm {rc}} [R_{{\rm E}}]$', fontsize=label_size, color=label_color,
                    labelpad=label_pad)
    axs3.set_ylabel(y_label, fontsize=label_size, color=label_color, labelpad=label_pad)

    # Plot the histogram of the bisection data
    axs4 = plt.subplot(gs[1, 1])
    axs4.hist(df_bisec.r_rc, bins=bins, range=(0, 15), color='#d62728', alpha=0.5, density=density,
              histtype=histtype, linewidth=linewidth)
    # Plot the median of the bisection data and add atext to the line
    axs4.axvline(bisec_r_rc_mean, color='#d62728', linestyle='--', linewidth=2)
    axs4.text(bisec_r_rc_mean+0.2, axs4.get_ylim()[1]*0.2,
              "$R_{{\\rm{{rc}}}}$ = {:.2f}$R_{{\\rm E}}$".format(bisec_r_rc_mean),
              fontsize=1.1*t_label_size, color=label_color)
    axs4.text(0.05, 0.9, "(d)", fontsize=1.1*t_label_size, color=label_color,
              transform=axs4.transAxes)
    axs4.set_xlim(r_lim[0], r_lim[1])
    axs4.set_xscale('linear')
    axs4.set_xlabel(r'$R_{\rm {rc}} [R_{{\rm E}}]$', fontsize=label_size, color=label_color,
                    labelpad=label_pad)
    axs4.set_ylabel(y_label, fontsize=label_size, color=label_color, labelpad=label_pad)
    axs4.yaxis.set_label_position("right")

    # Set the tick parameters
    axs1.tick_params(axis='both', direction='in', which='major', left=True, right=True,
                     top=True, bottom=True, labelleft=True, labelright=False,
                     labeltop=False, labelbottom=False, labelsize=t_label_size,
                     length=tick_len, width=tick_width, labelcolor=label_color)

    axs2.tick_params(axis='both', direction='in', which='major', left=True, right=True, top=True,
                     bottom=True, labelleft=False, labelright=True, labeltop=False,
                     labelbottom=False, labelsize=t_label_size, length=tick_len, width=tick_width,
                     labelcolor=label_color)

    axs3.tick_params(axis='both', direction='in', which='major', left=True, right=True, top=True,
                     bottom=True, labelleft=True, labelright=False, labeltop=False,
                     labelbottom=True, labelsize=t_label_size, length=tick_len, width=tick_width,
                     labelcolor=label_color)

    axs4.tick_params(axis='both', direction='in', which='major', left=True, right=True, top=True,
                     bottom=True, labelleft=False, labelright=True, labeltop=False,
                     labelbottom=True, labelsize=t_label_size, length=tick_len,
                     width=tick_width, labelcolor=label_color)

    axs1.text(0.95, .95, 'Shear', ha='right', va='top',
              transform=axs1.transAxes, fontsize=c_label_size, color=label_color)
    axs2.text(0.95, .95, 'Reconnection\n Energy', ha='right', va='top',
              transform=axs2.transAxes, fontsize=c_label_size, color=label_color)
    axs3.text(0.95, .95, 'Exhaust\n Velocity', ha='right', va='top',
              transform=axs3.transAxes, fontsize=c_label_size, color=label_color)
    axs4.text(0.95, .95, 'Bisection', ha='right', va='top',
              transform=axs4.transAxes, fontsize=c_label_size, color=label_color)

    # Show minor ticks
    axs1.minorticks_on()
    axs1.tick_params(axis='both', which='minor', direction='in', length=mtick_len, left=True,
                     right=True, top=True, bottom=True, color=mtick_color, width=mtick_width)
    axs2.minorticks_on()
    axs2.tick_params(axis='both', which='minor', direction='in', length=mtick_len, left=True,
                     right=True, top=True, bottom=True, color=mtick_color, width=mtick_width)
    axs3.minorticks_on()
    axs3.tick_params(axis='both', which='minor', direction='in', length=mtick_len, left=True,
                     right=True, top=True, bottom=True, color=mtick_color, width=mtick_width)
    axs4.minorticks_on()
    axs4.tick_params(axis='both', which='minor', direction='in', length=mtick_len, left=True,
                     right=True, top=True, bottom=True, color=mtick_color, width=mtick_width)

    # Set the number of ticks on the x-and y-axis
    axs1.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
    axs1.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
    axs2.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
    axs2.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
    axs3.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
    axs3.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
    axs4.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
    axs4.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))

    # Setting the tickmarks labels in such a way that they don't overlap
    plt.setp(axs1.get_xticklabels(), rotation=0, ha='right', va='top', visible=True)
    plt.setp(axs1.get_yticklabels(), rotation=0, va='center', visible=True)

    if dark_mode:
        transparent = False
    else:
        transparent = True

    # Save the figure
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    fig_name = f"{fig_folder}/{fig_name}_{cut_type}.{fig_format}"
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.05,
                dpi=200, transparent=transparent, format=fig_format)
    plt.close()
    print(f"Figure saved as {fig_name} in {fig_format} format in {fig_folder}")

    return df_shear, df_rx_en, df_va_cs, df_bisec
