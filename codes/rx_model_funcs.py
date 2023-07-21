import datetime
import multiprocessing as mp
import os
import warnings

import geopack.geopack as gp
import h5py as hf
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspedas as spd
import pytplot as ptt
import scipy as sp
import trjtrypy as tt
from dateutil import parser
from matplotlib.pyplot import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.filters import frangi
from tabulate import tabulate


def get_shear(b_vec_1, b_vec_2, angle_unit="radians"):
    r"""
    Get the shear angle between two magnetic field lines.

    Parameters
    ----------
    b_vec_1 : array of shape 1x3
        Input magnetic field vector.
    b_vec_2 : array of shape 1x3
        Input magnetic field vector.
    angle_unit : str, optional
        Preferred unit of angle returned by the code. Default is "radians".

    Raises
    ------
    KeyError If the key is not input_angle is not set to "radians" or "degrees" then the code raises
        a key error.

    Returns
    -------
    angle: float
        Angle between the two vectors, in radians by default
    """
    unit_vec_1 = b_vec_1 / np.linalg.norm(b_vec_1)
    unit_vec_2 = b_vec_2 / np.linalg.norm(b_vec_2)
    angle = np.arccos(np.dot(unit_vec_1, unit_vec_2))

    if (angle_unit == "radians"):
        return angle
    elif (angle_unit == "degrees"):
        return angle * 180 / np.pi
    else:
        raise KeyError("angle_unit must be radians or degrees")


def get_rxben(b_vec_1, b_vec_2):
    r"""
    Get the reconnection energy between two magnetic field lines.

    It has the following mathematical expression:

    .. math:: rexben = 0.5 (|\vec{B_1}| + \vec{B_2}) (1 - \hat{B_1} \cdot \hat{B_2})

    Parameters
    ----------
    b_vec_1 : array of shape 1x3
        Input magnetic field vector.
    b_vec_2 : array of shape 1x3
        Input magnetic field vector.

    Returns
    -------
    rxben : float
        Reconnection field energy density in nPa
    """

    # alpha = - 14.87 * np.pi / 180  # radians (From Hesse2013)
    b_vec_1 = np.array(b_vec_1)
    b_vec_2 = np.array(b_vec_2)
    mag_vec_1 = np.linalg.norm(b_vec_1)
    mag_vec_2 = np.linalg.norm(b_vec_2)

    unit_vec_1 = b_vec_1 / mag_vec_1
    unit_vec_2 = b_vec_2 / mag_vec_2

    # The bisector vector of thwo input vectors
    unit_vec_bisec = (unit_vec_1 + unit_vec_2) / np.linalg.norm(unit_vec_1 + unit_vec_2)

    # Cross product of the two input vectors with the bisector vector to get the reconnection
    # component of the magnetic field.
    rx_b_1 = np.cross(b_vec_1, unit_vec_bisec)
    rx_b_2 = np.cross(b_vec_2, unit_vec_bisec)

    rx_b_mag_1 = np.linalg.norm(rx_b_1)
    rx_b_mag_2 = np.linalg.norm(rx_b_2)

    # Reconnection energy (from Hesse2013)
    rx_en = rx_b_mag_1 ** 2 * rx_b_mag_2 ** 2

    return rx_en


def get_vcs(b_vec_1, b_vec_2, n_1, n_2):
    r"""
    Get vcs code.

    Parameters
    ----------
    b_vec_1 : array of shape 1x3
        Input magnetic field vector.
    b_vec_2 : array of shape 1x3
        Input magnetic field vector.
    n_1 : float
        Density of first component
    n_2 : float
        Density of second component

    Returns
    -------
    vcs : float
        The exhaust velocity in km/s
    """
    va_p1 = 21.812  # conv. nT, m_P/cm ^ 3 product to km/s cassak-shay

    mag_vec_1 = np.linalg.norm(b_vec_1)
    mag_vec_2 = np.linalg.norm(b_vec_2)

    unit_vec_1 = b_vec_1 / mag_vec_1
    unit_vec_2 = b_vec_2 / mag_vec_2

    # The bisector vector of thwo input vectors
    unit_vec_bisec = (unit_vec_1 + unit_vec_2) / np.linalg.norm(unit_vec_1 + unit_vec_2)

    # Cross product of the two input vectors with the bisector vector to get the reconnection
    # component of the magnetic field.
    rx_b_1 = np.cross(b_vec_1, unit_vec_bisec)
    rx_b_2 = np.cross(b_vec_2, unit_vec_bisec)

    # Magnitude of the reconnection component of the magnetic fields
    rx_mag_1 = np.linalg.norm(rx_b_1)
    rx_mag_2 = np.linalg.norm(rx_b_2)

    vcs = va_p1 * np.sqrt(rx_mag_1 * rx_mag_2 * (rx_mag_1 +
                          rx_mag_2) / (rx_mag_1 * n_2 + rx_mag_2 * n_1))

    return vcs


def get_bis(b_vec_1, b_vec_2):
    r"""
    Get the shear angle between two magnetic field lines.

    Parameters
    ----------
    b_vec_1 : array of shape 1x3
        First input magnetic field vector.
    b_vec_2 : array of shape 1x3
        Second input magnetic field vector.

    Raises
    ------
    KeyError If the key is not input_angle is not set to "radians" or "degrees" then the code raises
        a key error.

    Returns
    -------
    bis_field_1 : float
        The magnitude of bisected field line corresponding to the first input magnetic field vector.

    bis_field_2 : float
        The magnitude of bisected field line corresponding to the second input magnetic field
        vector.
    """
    b_vec_1 = np.array(b_vec_1)
    b_vec_2 = np.array(b_vec_2)

    mag_vec_1 = np.linalg.norm(b_vec_1)
    mag_vec_2 = np.linalg.norm(b_vec_2)

    unit_vec_1 = b_vec_1 / mag_vec_1
    unit_vec_2 = b_vec_2 / mag_vec_2

    unit_vec_bisect = (unit_vec_1 + unit_vec_2) / np.linalg.norm(unit_vec_1 + unit_vec_2)

    # Cross product of the two vectors with the bisector
    b_vec_1_cross = np.cross(b_vec_1, unit_vec_bisect)
    b_vec_2_cross = np.cross(b_vec_2, unit_vec_bisect)

    # Magnitude of the cross product vectors (should be the same for both for a symmetric
    # reconnection). These magnitude are the reconnecting components of the magnetic field.
    bis_field_1 = np.linalg.norm(b_vec_1_cross)
    bis_field_2 = np.linalg.norm(b_vec_2_cross)
    return bis_field_1, bis_field_2


def get_ca(b_vec, angle_unit="radians"):
    r"""
    Get ca.

    Parameters
    ----------
    b_vec : array of shape 1x3
        Input magnetic field vector.
    angle_unit : str, optional
        Preferred unit of angle returned by the code. Default is "radians".

    Raises
    ------
    KeyError If the key is not input_angle is not set to "radians" or "degrees" then the code raises
        a key error.

    Returns
    -------
    angle : float Returns arctan of y- and z-component of the magnetic field vector.

    """
    angle = np.arctan(b_vec[1] / b_vec[2])

    if (angle_unit == "radians"):
        return angle
    elif (angle_unit == "degrees"):
        return angle * 180 / np.pi
    else:
        raise KeyError("angle_unit must be radians or degrees")


def ridge_finder_multiple(
    image=[None, None, None, None],
    convolution_order=[1, 1, 1, 1],
    t_range=["2016-12-24 15:08:00", "2016-12-24 15:12:00"],
    dt=5,
    b_imf=[-5, 0, 0],
    b_msh=[-5, 0, 0],
    v_msh=[-200, 50, 50],
    xrange=[-15.1, 15],
    yrange=[-15.1, 15],
    mms_probe_num="1",
    mms_sc_pos=[0, 0],
    dr=0.5,
    dipole_tilt_angle=None,
    p_dyn=None,
    imf_clock_angle=None,
    sigma=[2.2, 2.2, 2.2, 2.2],
    mode="nearest",
    alpha=1.,
    vmin=[None, None, None, None],
    vmax=[None, None, None, None],
    cmap_list=["viridis", "viridis", "viridis", "viridis"],
    draw_patch=[True, True, True, True],
    draw_ridge=[True, True, True, True],
    save_fig=True,
    fig_name="new",
    fig_format="png",
    c_label=[None, None, None, None],
    wspace=0.1,
    hspace=0.1,
    fig_size=(10, 10),
    box_style=None,
    title_y_pos=0.95,
    interpolation="nearest",
    tsy_model="t96",
    dark_mode=True,
    rc_file_name="rc_file.csv",
    rc_folder="../data",
    save_rc_file=False,
    fig_version="v001",
    df_jet_reversal=None,
):
    r"""
    Finds ridges in an image and plot the points with maximum ridge value on the given image.

    Parameters
    ----------
    image : list of numpy arrays
        List of images to be plotted.
    convolution_order : list of ints
        List of the order of the convolution to be used while smoothing each image. Values must be
        non-negative integers. Default is [1, 1, 1, 1].
    t_range : list of str
            The time range to find the ridge in. Default is ["2016-12-24 15:08:00",
            "2016-12-24 15:12:00"].
    dt : float, optional
        The time differential, in minutes, for observation if "t_range" has only one element.
        Default is 5 minutes.
    b_imf : list of floats, optional
            The IMF vector. Default is [-5, 0, 0].
    b_msh : list of floats, optional
            The MSH vector. Default is [-5, 0, 0].
    v_msh : list of floats, optional
            The MSH velocity vector. Default is [-200, 50, 50].
    xrange : list of floats, optional
            The range of x-values for image. Default is [-15.1, 15].
    yrange : list of floats, optional
            The range of y-values for image. Default is [-15.1, 15].
    mms_probe_num : str, optional
            The probe number of the MMS spacecraft. Default is 1.
    mms_sc_pos : list of floats, optional
            The position of the spacecraft in the image. Default is [0, 0].
    dr : float, optional
            The step size for the grid. Default is 0.5.
    dipole_tilt_angle : float, optional
            The dipole tilt angle. Default is None.
    p_dyn : float, optional
            The dynamic pressure. Default is None.
    imf_clock_angle : float, optional
            The IMF clock angle. Default is None.
    sigma : list of floats, optional
            List of sigmas to be used for the ridge plot. Default is [2.2, 2.2, 2.2, 2.2].
    mode : str
            The mode of the filter. Can be "nearest", "reflect", "constant", "mirror", "wrap" or
            "linear". Default is "nearest".
    alpha : float
            The alpha value for the filter. Default is 0.5.
    vmin : list of floats, optional
            List of vmin values for the ridge plot. Default is [None, None, None, None].
    vmax : list of floats, optional
            List of vmax values for the ridge plot. Default is [None, None, None, None].
    cmap_list : list of str, optional
            List of colormaps to be used for the ridge plot. Default is ["viridis", "viridis",
            "viridis", "viridis"].
    draw_patch : list of bool, optional
            Whether to draw the circular patch. Default is [True, True, True, True].
    draw_ridge : list of bool, optional
            Whether to draw the ridge line. Default is [True, True, True, True].
    save_fig : bool, optional
            Whether to save the figure. Default is True.
    fig_name : str, optional
            The name of the figure. Default is "new".
    fig_format : str, optional
            The format of the figure. Default is "png".
    c_label : list of str, optional
            List of colorbar labels. Default is [None, None, None, None].
    wspace : float, optional
            The width space between subplots. Default is 0.1.
    hspace : float, optional
            The height space between subplots. Default is 0.1.
    fig_size : tuple of floats, optional
            The size of the figure. Default is (10, 10).
    box_style : dict, optional
            The style of the box. Default is None.
    title_y_pos : float, optional
            The y-position of the title. Default is 0.95.
    interpolation : str, optional
            The interpolation method for imshow. Default is "nearest".
            Options are "nearest", "bilinear", "bicubic", "spline16", "spline36", "hanning",
            "hamming", "hermite", "kaiser", "quadric", "catrom", "gaussian", "bessel", "mitchell"
    tsy_model : str, optional
            The Tsyganenko model to be used. Default is "t96".
    dark_mode : bool, optional
        Sets the dark mode for the plot and adjusts the color of labels and tickmarks accordingly.
        Default is True.
    rc_file_name : str, optional
        The name of the file to save the reconnection line data. Default is "rc_file.csv".
    rc_folder : str, optional
        The folder to save the reconnection line data. Default is "../data".
    save_rc_file : bool, optional
        Whether to save the reconnection line data. Default is False.
    fig_version : str, optional
        The version of the figure. Default is "v001".
    df_jet_reversal : pandas DataFrame, optional
        The dataframe containing the jet reversal times. Default is None.

    Raises
    ------
    ValueError: If the image is not a numpy array.

    Returns
    -------
    ridge_points : ndarray
    """
    if image is None:
        raise ValueError("No image given")

    if len(t_range) == 1:
        # Check if t_range is a datetime object
        if isinstance(t_range[0], datetime.datetime):
            t_range_date = t_range[0]
        else:
            t_range_date = datetime.datetime.strptime(t_range[0], "%Y-%m-%d %H:%M:%S")
        t_range_date_min = t_range_date - datetime.timedelta(minutes=dt)
        t_range_date_max = t_range_date + datetime.timedelta(minutes=dt)
        t_range = [t_range_date_min.strftime("%Y-%m-%d %H:%M:%S"),
                   t_range_date_max.strftime("%Y-%m-%d %H:%M:%S")]

    if dark_mode:
        plt.style.use("dark_background")
        # tick_color = "w"  # color of the tick lines
        mtick_color = "w"  # color of the minor tick lines
        label_color = "w"  # color of the tick labels
        clabel_color = "w"  # color of the colorbar label
    else:
        plt.style.use("default")
        # tick_color = "k"  # color of the tick lines
        mtick_color = "k"  # color of the minor tick lines
        label_color = "k"  # color of the tick labels
        clabel_color = "k"  # color of the colorbar label

    # Set the fontstyle to Times New Roman
    font = {"family": "serif", "weight": "normal", "size": 10}
    plt.rc("font", **font)
    plt.rc("text", usetex=True)
    fig = plt.figure(num=None, figsize=fig_size, dpi=200, facecolor="w", edgecolor="k")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=wspace, hspace=hspace)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])

    # Set the font size for the axes
    label_size = 20  # fontsize for x and y labels
    t_label_size = 18  # fontsize for tick label
    c_label_size = 18  # fontsize for colorbar label
    ct_tick_size = 14  # fontsize for colorbar tick labels
    l_label_size = 14  # fontsize for legend label

    tick_len = 10  # length of the tick lines
    mtick_len = 7  # length of the minor tick lines
    tick_width = 1  # tick width in points
    mtick_width = 0.7  # minor tick width in points

    # box_style = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    if dark_mode:
        box_style = box_style
    else:
        box_style = dict(boxstyle="round", color="w", alpha=0.8, linewidth=1)
    y_vals = []
    x_intr_vals_list = []
    y_intr_vals_list = []
    for i in range(len(image)):
        image_rotated = np.transpose(image[i])

        # Create the masked image from result for all the new processings
        # Find the number of rows in the original image
        n_rows, n_cols = image_rotated.shape

        # Make a grid of the data based on mumber of rows and columns
        X, Y = np.ogrid[:n_rows, :n_cols]

        # Find the central row and column
        c_row = int(n_rows/2)
        c_col = int(n_cols/2)
        # Find the distance of each pixel from the central pixel in terms of pixels
        dist_pxl = np.sqrt((X - c_row) ** 2 + (Y - c_col) ** 2)
        mask_image = dist_pxl > xrange[1] / dr

        if cmap_list is None:
            cmap_list = ["viridis", "viridis", "viridis", "viridis"]
        else:
            cmap_list = cmap_list
        if (vmin is not None and vmax is not None):
            norm = plt.Normalize(vmin=vmin[i], vmax=vmax[i])
        else:
            norm = plt.Normalize()

        kwargs = {"sigmas": [sigma[i]], "black_ridges": False, "mode": mode, "alpha": 1}

        # Smoothen the image
        image_smooth = sp.ndimage.gaussian_filter(image_rotated, order=convolution_order[i],
                                                  sigma=[5, 5], mode=mode)
        image_smooth_p = sp.ndimage.gaussian_filter(image_rotated, order=0, sigma=[5, 5],
                                                    mode=mode)
        result = frangi(image_smooth, **kwargs)  # frangi, hessian, meijering, sato

        m_result = result.copy()
        m_result[mask_image] = np.nan
        new_image_rotated = image_rotated.copy()
        new_image_rotated[mask_image] = np.nan

        x_len = image_rotated.shape[0]
        y_len = image_rotated.shape[1]

        y_val = np.full(y_len, np.nan)
        y_vals.append(y_val)
        im_max_val = np.full(y_len, np.nan)
        for xx in range(y_len):
            try:
                y_val[xx] = np.nanargmax(m_result[:, xx]) * dr + yrange[0]
                im_max_val[xx] = np.nanargmax(new_image_rotated[:, xx]) * dr + yrange[0]
            except Exception:
                pass

        # TODO: Find a better way to do this
        j = i//2
        k = i % 2

        axs1 = plt.subplot(gs[j, k])
        im1 = axs1.imshow(image_smooth_p, extent=[xrange[0], xrange[1], yrange[0], yrange[1]],
                          origin="lower", cmap=cmap_list[i], norm=norm, interpolation=interpolation,
                          alpha=1)
        divider1 = make_axes_locatable(axs1)
        # Draw a circle of radius 10 around the center of the image
        axs1.add_patch(plt.Circle((0, 0), radius=15, color="gray", fill=False, lw=0.5))

        # Take rolling average of the y_val array
        y_val_avg = np.full(len(y_val), np.nan)
        im_max_val_avg = np.full(len(y_val), np.nan)

        r_a_l = 5
        for xx in range(len(y_val)):
            y_val_avg[xx] = np.nanmean(y_val[max(0, xx - r_a_l):min(len(y_val), xx + r_a_l)])
            im_max_val_avg[xx] = np.nanmean(im_max_val[max(0, xx - r_a_l):min(len(y_val),
                                            xx + r_a_l)])

        if draw_ridge:
            x_intr_vals = np.linspace(xrange[0], xrange[1], x_len)
            y_intr_vals = im_max_val_avg
            axs1.plot(x_intr_vals, y_intr_vals, color="aqua", ls="-", alpha=0.9)

        # Find the interpolation function corresponding to the x_vals and y_val_avg array
        line_intrp = line_fnc_der(x=np.linspace(xrange[0], xrange[1], x_len), y=im_max_val_avg)

        # Spacecraft position
        r0 = mms_sc_pos[:3]

        x_vals = np.linspace(xrange[0], xrange[1], x_len)

        curve = np.array([[x_vals[i], im_max_val_avg[i]] for i in range(len(x_vals))])
        points = np.array([r0[1:]])
        curves = np.array([curve], dtype=object)

        # compute unsigned distance
        dist_u = tt.basedists.distance(points, curves, argPnts=True)

        # Direction of the magnetosheath magnetic field at the position of the spacecraft
        # TODO: Check if this is same as direction/magnitude given by the Cooling model
        b_msh_dir = b_msh[:3] / np.linalg.norm(b_msh[:3])
        v_msh_dir = v_msh[:3] / np.linalg.norm(v_msh[:3])

        # Find the closest point on the reconnection line in the direction of the magnetosheath
        # magnetic field.
        xn = np.full(300, np.nan)
        yn = np.full(300, np.nan)
        for n in range(-150, 150):
            xn[50 + n] = r0[1] + n / 3 * b_msh_dir[1]
            yn[50 + n] = r0[2] + n / 3 * b_msh_dir[2]

        # Find the expected values of y-coordinate based on the x-coordinate, in the direction of
        # the magnetosheath magnetic field.
        yn_interp = line_intrp(xn)

        # Compute the distance between the spacecraft position and the coordinates computed in
        # previous step.
        dist_rn = np.abs(yn - yn_interp)
        # Find the index of the minimum distance
        min_dist_rn_idx = np.argmin(dist_rn)

        # Find the x- and y-coordinates corresponding to the minimum distance
        xn_rc = xn[min_dist_rn_idx]
        yn_rc = yn[min_dist_rn_idx]

        # Save the x- and y-coordinates of the reconnection line along with the spacecraft position
        x_intr_vals = [r0[1], xn_rc]
        y_intr_vals = [r0[2], yn_rc]

        # Append the x- and y-coordinates to the list in order to save to a file
        x_intr_vals_list.append(x_intr_vals)
        y_intr_vals_list.append(y_intr_vals)

        # Find the distance between the spacecraft position and the reconnection line
        dist_rc = dist_u[0]["UnsignedDistance"][0]
        x_y_point = dist_u[0]["ArgminPoints"][0]

        if dist_rc > xrange[1]:
            dist_rc = np.nan

        if i == 0:
            method_used = "shear"
        elif i == 1:
            method_used = "rx_en"
        elif i == 2:
            method_used = "va_cs"
        elif i == 3:
            method_used = "bisection"

        # Save the data to a text file
        # Create the rc-folder if it doesn"t exist
        if save_rc_file:
            if not os.path.exists(rc_folder):
                os.makedirs(rc_folder)
            var_list = "mms_spc_num,date_from,date_to,spc_pos_x,spc_pos_y,spc_pos_z,"\
                       "b_msh_x,b_msh_y,b_msh_z,r_rc,method_used,b_imf_x,b_imf_y,"\
                       "b_imf_z,dipole,imf_clock_angle,p_dyn"
            data_dict = {
                "mms_spc_num": mms_probe_num,
                "date_from": t_range[0],
                "date_to": t_range[1],
                "spc_pos_x": r0[0],
                "spc_pos_y": r0[1],
                "spc_pos_z": r0[2],
                "b_msh_x": b_msh[0],
                "b_msh_y": b_msh[1],
                "b_msh_z": b_msh[2],
                "r_rc": np.round(dist_rc, 3),
                "method_used": method_used,
                "b_imf_x": b_imf[0],
                "b_imf_y": b_imf[1],
                "b_imf_z": b_imf[2],
                "dipole": dipole_tilt_angle * 180 / np.pi,
                "imf_clock_angle": imf_clock_angle,
                "p_dyn": p_dyn
            }
            # Add keys and data from df_jet_reversal to data_dict if those keys aren"t already
            # present in the dictionary
            try:
                for key in df_jet_reversal.keys():
                    if key not in data_dict.keys():
                        data_dict[key] = df_jet_reversal[key]
                        # Add the key to the variable list
                        var_list += "," + key
            except Exception:
                pass
            # Save data to the csv file using tab delimiter
            if not os.path.exists(rc_folder + rc_file_name):
                with open(rc_folder + rc_file_name, "w") as f:
                    f.write(var_list + "\n")
                    f.close()
                    print(f"Created {rc_folder + rc_file_name} to store data")
            # Open file and append the relevant data
            with open(rc_folder + rc_file_name, "a") as f:
                for key in data_dict.keys():
                    try:
                        f.write(f"{np.round(data_dict[key], 3)},")
                    except Exception:
                        f.write(f"{data_dict[key]},")
                f.write("\n")
                f.close()
                print(f"Saved data to {rc_folder + rc_file_name}")

        # plot an arrow along the magnetosheath magnetic field direction
        axs1.arrow(r0[1], r0[2], 5 * b_msh_dir[1], 5 * b_msh_dir[2], head_width=0.4,
                   head_length=0.7, fc="w", ec="r", linewidth=2, ls="-")

        # Plot the arrow along the magnetosheath velocity direction
        axs1.arrow(r0[1], r0[2], 5 * v_msh_dir[1], 5 * v_msh_dir[2], head_width=0.4,
                   head_length=0.7, fc="w", ec="b", linewidth=2, ls="-")

        # Plot line connecting the spacecraft position and the reconnection line
        if ~np.isnan(dist_rc):
            # axs1.plot(x_intr_vals, y_intr_vals, "--", color="w", linewidth=2)
            axs1.plot([r0[1], x_y_point[0]], [r0[2], x_y_point[1]], "--", color="w", linewidth=2)
            distance = f"$R_{{\\rm rc}}$ = {dist_rc:.2f} $R_{{\\rm E}}$"
            axs1.text(x_intr_vals[0] - 6, y_intr_vals[0] + 2, distance, fontsize=l_label_size * 1.2,
                      color="k", ha="left", va="bottom")

        # Plot a horizontal line at x=0 and a vertical line at y=0
        axs1.axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
        axs1.axvline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)

        if (draw_patch):
            patch = patches.Circle((0, 0), radius=xrange[1], transform=axs1.transData, fc="none",
                                   ec="k", lw=0.5)
            im1.set_clip_path(patch)
        axs1.add_patch(patch)
        if i == 0 or i == 2:
            axs1.set_ylabel(r"Z [GSM, $R_{\rm E}$]", fontsize=label_size, color=label_color)
        if i == 2 or i == 3:
            axs1.set_xlabel(r"Y [GSM, $R_{\rm E}$]", fontsize=label_size, color=label_color)
        if i == 1 or i == 3:
            axs1.set_ylabel(r"Z [GSM, $R_{\rm E}$]", fontsize=label_size, color=label_color)
            axs1.yaxis.set_label_position("right")

        if dark_mode:
            text_color = "white"
        else:
            text_color = "black"
        if i == 0:
            axs1.text(-0.15, 1.16, f"Model: {tsy_model}", horizontalalignment="left",
                      verticalalignment="bottom", transform=axs1.transAxes, rotation=0,
                      color=text_color, fontsize=l_label_size, bbox=box_style)

        if i == 1:
            axs1.text(1.15, 1.16, f"MMS Position [GSM] - {mms_sc_pos[0], mms_sc_pos[1], mms_sc_pos[2]}",
                      horizontalalignment="right",
                      verticalalignment="bottom", transform=axs1.transAxes, rotation=0,
                      color=text_color, fontsize=l_label_size, bbox=box_style)

        # Define the location of the colorbar, it"s size relative to main figure and the padding
        # between the colorbar and the figure, the orientation the colorbar
        cax1 = divider1.append_axes("top", size="5%", pad=0.01)
        cbar1 = plt.colorbar(im1, cax=cax1, orientation="horizontal", ticks=None, fraction=0.05,
                             pad=0.01)
        cbar1.ax.tick_params(axis="x", direction="in", top=True, labeltop=True, bottom=False,
                             labelbottom=False, pad=0.01, labelsize=ct_tick_size,
                             labelcolor=label_color)
        cbar1.ax.xaxis.set_label_position("top")
        cbar1.ax.set_xlabel(f"{c_label[i]}", fontsize=c_label_size,
                            color=clabel_color)
        # Draw the spacecraft position
        axs1.plot(mms_sc_pos[1], mms_sc_pos[2], "white", marker="$\\bigoplus$", ms=15, alpha=1)

        # Set tick label parameters
        if i == 0 or i == 2:
            axs1.tick_params(axis="both", direction="in", which="major", left=True, right=True,
                             top=True, bottom=True, labelleft=True, labelright=False,
                             labeltop=False, labelbottom=True, labelsize=t_label_size,
                             length=tick_len, width=tick_width, labelcolor=label_color)
        else:
            axs1.tick_params(axis="both", direction="in", which="major", left=True, right=True,
                             top=True, bottom=True, labelleft=False, labelright=True,
                             labeltop=False, labelbottom=True, labelsize=t_label_size,
                             length=tick_len, width=tick_width, labelcolor=label_color)
        # Write the timme range on the plot
        if i == 2:
            axs1.text(-0.17, -0.1, f"Clock Angle: {np.round(imf_clock_angle, 2)}$^\\circ$",
                      horizontalalignment="left", verticalalignment="top", transform=axs1.transAxes,
                      rotation=0, color=text_color, fontsize=l_label_size, bbox=box_style)
        elif i == 3:
            axs1.text(1.17, -0.1,
                      f"Dipole tilt: {np.round(dipole_tilt_angle * 180/np.pi, 2)} ${{\\hspace{{-.2em}}}}^\\circ$",
                      horizontalalignment="right", verticalalignment="top",
                      transform=axs1.transAxes, rotation=0, color=text_color, fontsize=l_label_size,
                      bbox=box_style)
        if i == 0:
            # Add a label "(a)" to the plot to indicate the panel number
            axs1.text(0.05, 0.1, "(a)", horizontalalignment="left", verticalalignment="top",
                      transform=axs1.transAxes, rotation=0, color=text_color,
                      fontsize=1.2 * l_label_size)
        elif i == 1:
            # Add a label "(b)" to the plot to indicate the panel number
            axs1.text(0.1, 0.1, "(b)", horizontalalignment="right", verticalalignment="top",
                      transform=axs1.transAxes, rotation=0, color=text_color,
                      fontsize=1.2 * l_label_size)
        elif i == 2:
            # Add a label "(c)" to the plot to indicate the panel number
            axs1.text(0.05, 0.1, "(c)", horizontalalignment="left", verticalalignment="top",
                      transform=axs1.transAxes, rotation=0, color=text_color,
                      fontsize=1.2 * l_label_size)
        elif i == 3:
            # Add a label "(d)" to the plot to indicate the panel number
            axs1.text(0.1, 0.1, "(d)", horizontalalignment="right", verticalalignment="top",
                      transform=axs1.transAxes, rotation=0, color=text_color,
                      fontsize=1.2 * l_label_size)

        # Show minor ticks
        axs1.minorticks_on()
        axs1.tick_params(axis="both", which="minor", direction="in", length=mtick_len, left=True,
                         right=True, top=True, bottom=True, color=mtick_color, width=mtick_width)
        # Set the number of ticks on the x-axis
        axs1.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
        # Set the number of ticks on the y-axis
        axs1.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))

        # Setting the tickmarks labels in such a way that they don"t overlap
        plt.setp(axs1.get_xticklabels(), rotation=0, ha="right", va="top", visible=True)
        plt.setp(axs1.get_yticklabels(), rotation=0, va="center", visible=True)
        # Set the title of the plot
        fig.suptitle(f"Time range: {t_range[0]} - {t_range[1]} \n $B_{{\\rm {{imf}}}}$ = {b_imf[0], b_imf[1], b_imf[2]}",
                     fontsize=label_size, color=text_color, y=title_y_pos, alpha=0.65)

    if save_fig:
        try:
            # TODO: Add folder name as one of the path and make sure that the code creates the
            # folder. Gives out error if the folder can"t be created.
            temp1 = parser.parse(t_range[1]).strftime("%Y-%m-%d_%H-%M-%S")
            fig_time_range = f"{parser.parse(t_range[0]).strftime('%Y-%m-%d_%H-%M-%S')}_{temp1}"
            fig_folder = f"../figures/all_ridge_plots/{tsy_model}/{interpolation}" +\
                         f"_interpolation_mms{mms_probe_num}/{fig_version}"
            check_folder = os.path.isdir(fig_folder)
            # If folder doesn"t exist, then create it.
            if not check_folder:
                os.makedirs(fig_folder)
                print("created folder : ", fig_folder)
            else:
                print(f"folder already exists: {fig_folder}\n")

            bbb = f"{b_imf[0]:.0f}_{b_imf[1]:.0f}_{b_imf[2]:.0f}"
            fig_name = f"{fig_folder}/ridge_plot_{fig_time_range}_{bbb}.{fig_format}"
            plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.05, format=fig_format, dpi=200)
            print(f"Figure saved as {fig_name}")
        except Exception as e:
            print(e)
            print("Figure not saved, folder does not exist. Create folder ../figures")
            # pass
        # plt.close()
    plt.close()
    return y_vals, x_intr_vals_list, y_intr_vals_list


def model_run(*args):
    """
    Returns the value of the magnetic field at a given point in the model grid using three different
    models
    """
    j = args[0][0]
    k = args[0][1]
    y_max = args[0][2]
    z_max = args[0][3]
    dr = args[0][4]
    m_p = args[0][5]
    ro = args[0][6]
    alpha = args[0][7]
    rmp = args[0][8]
    sw_params = args[0][9]
    model_type = args[0][-1]

    y0 = int(j * dr) - y_max
    z0 = int(k * dr) - z_max
    rp = np.sqrt(y0**2 + z0**2)  # Projection of r into yz-plane

    d_theta = np.pi / 100

    for index in range(0, 100):

        theta = index * d_theta
        r = ro * (2 / (1 + np.cos(theta))) ** alpha
        zp = r * np.sin(theta)  # not really in z direction, but a distance in yz plane
        x0 = r * np.cos(theta)

        if y0 == 0:
            signy = 1.0
        else:
            signy = np.sign(y0)

        if z0 == 0:
            signz = 1.0
        else:
            signz = np.sign(z0)

        if (rp <= zp):

            x_shu = (r - m_p) * np.cos(theta)
            phi = np.arctan2(z0, y0)

            if (abs(y0) == 0 or abs(z0) == 0):
                if (abs(y0) == 0):
                    y_shu = 0
                    z_shu = (r - m_p) * np.sin(theta)
                elif (abs(z0) == 0):
                    z_shu = 0
                    y_shu = (r - m_p) * np.sin(theta)
            else:
                z_shu = np.sqrt((rp - 1.0)**2 / (1 + np.tan(phi)**(-2)))
                y_shu = z_shu / np.tan(phi)

            m_proton = 1.672e-27  # Mass of proton in SI unit
            n_sh = sw_params["rho"] * (1.509 * np.exp(x_shu / rmp) + .1285) / m_proton

            y_shu = abs(y_shu) * signy
            z_shu = abs(z_shu) * signz

            # Cooling JGR 2001 Model, equation 9 to 12
            # the distance from the focus to the magnetopause surface
            A = 2
            ll = 3 * rmp / 2 - x0
            b_msx = - A * (- sw_params["b_imf"][0] * (1 - rmp / (2 * ll)) + sw_params["b_imf"][1]
                           * (y0 / ll) + sw_params["b_imf"][2] * (z0 / ll))
            b_msy = A * (- sw_params["b_imf"][0] * (y0 / (2 * ll)) + sw_params["b_imf"][1]
                         * (2 - y0**2 / (ll * rmp)) - sw_params["b_imf"][2] * (y0 * z0 / (ll *
                                                                                          rmp)))
            b_msz = A * (- sw_params["b_imf"][0] * (z0 / (2 * ll)) - sw_params["b_imf"][1]
                         * (y0 * z0 / (ll * rmp)) + sw_params["b_imf"][2] * (2 - z0**2 / (ll *
                                                                                          rmp)))
            try:
                if model_type == "t96":
                    bx_ext, by_ext, bz_ext = gp.t96.t96(sw_params["param"], sw_params["ps"], x_shu,
                                                        y_shu, z_shu)
                elif model_type == "t01":
                    bx_ext, by_ext, bz_ext = gp.t01.t01(sw_params["param"], sw_params["ps"], x_shu,
                                                        y_shu, z_shu)
            except Exception:
                print(f"Skipped for {x_shu, y_shu, z_shu}")
                pass

            bx_igrf, by_igrf, bz_igrf = gp.igrf_gsm(x_shu, y_shu, z_shu)

            # print(j, k, bx_ext, bx_igrf)
            bx = bx_ext + bx_igrf
            by = by_ext + by_igrf
            bz = bz_ext + bz_igrf

            shear = get_shear([bx, by, bz], [b_msx, b_msy, b_msz], angle_unit="degrees")

            rx_en = get_rxben([bx, by, bz], [b_msx, b_msy, b_msz])
            va_cs = get_vcs([bx, by, bz], [b_msx, b_msy, b_msz], n_sh, 0.1)
            bisec_msp, bisec_msh = get_bis([bx, by, bz], [b_msx, b_msy, b_msz])
            break

    return (j, k, bx, by, bz, shear, rx_en, va_cs, bisec_msp, bisec_msh, x_shu, y_shu, z_shu, b_msx,
            b_msy, b_msz)


def get_sw_params(
    probe=None,
    omni_level="hro",
    time_clip=True,
    trange=None,
    mms_probe_num=None,
    verbose=False
):
    r"""
    Get the solar wind parameters from the OMNI database.

    Parameters
    ----------
    probe : str
        The probe to use. Default is "None".
    omni_level : str
        The omni data level to use. Options are "hro" and "hro2". Default is "hro".
    time_clip : bool
        If True, the data will be clipped to the time range specified by trange. Default is True.
    trange : list or an array of length 2
        The time range to use. Should in the format [start, end], where start and end times should
        be a string in the format "YYYY-MM-DD HH:MM:SS".
    mms_probe_num : str
        The MMS probe to use. Options are "1", "2", "3" and "4". Default is None.
    verbose : bool
        If True, print out a few messages and the solar wind parameters. Default is False.

    Raises
    ------
    ValueError: If the probe is not one of the options.
    ValueError: If the trange is not in the correct format.

    Returns
    -------
    sw_params : dict
        The solar wind parameters.
    """

    if trange is None:
        raise ValueError("trange must be specified as a list of start and end times in the format"
                         "'YYYY-MM-DD HH:MM:SS'.")

    # Check if trange is either a list or an array of length 2
    if not isinstance(trange, (list, np.ndarray)) or len(trange) != 2:
        raise ValueError(
            "trange must be specified as a list or array of length 2 in the format" +
            "'YYYY-MM-DD HH:MM:SS'")

    # Download the OMNI data (default level of "hro_1min") for the specified timerange.
    omni_varnames = ["BX_GSE", "BY_GSM", "BZ_GSM", "proton_density", "Vx", "Vy", "Vz", "SYM_H", "T"]
    omni_vars = spd.omni.data(trange=trange, varnames=omni_varnames, level=omni_level,
                              time_clip=time_clip)

    omni_time = ptt.get_data(omni_vars[0])[0]
    # print(f"omni_time: {omni_time}")
    omni_bx_gse = ptt.get_data(omni_vars[0])[1]
    omni_by_gsm = ptt.get_data(omni_vars[1])[1]
    omni_bz_gsm = ptt.get_data(omni_vars[2])[1]
    omni_np = ptt.get_data(omni_vars[3])[1]
    omni_vx = ptt.get_data(omni_vars[4])[1]
    omni_vy = ptt.get_data(omni_vars[5])[1]
    omni_vz = ptt.get_data(omni_vars[6])[1]
    omni_sym_h = ptt.get_data(omni_vars[7])[1]
    omni_t_p = ptt.get_data(omni_vars[8])[1]

    # Convert omni_time to datetime objects from unix time
    omni_time_datetime = [datetime.datetime.utcfromtimestamp(t) for t in omni_time]
    # Get trange in datetime format
    omni_trange_time_object = [pd.to_datetime(trange[0]),
                               pd.to_datetime(trange[1])]
    # Add utc as timezone to omni_trange_time_object
    omni_trange_time_object = [t.replace(tzinfo=datetime.timezone.utc) for t in
                               omni_trange_time_object]

    # Get mms postion in GSM coordinates for the specified time range

    if (mms_probe_num is not None):
        # Define mms time as the center of the time range
        mms_time_t0 = pd.to_datetime(trange[0])
        mms_time_dt = pd.to_datetime(trange[1]) - mms_time_t0
        mms_time = mms_time_t0 + mms_time_dt / 2

        # Define mms time range as +/- 10 seconds from the mms time
        mms_trange = [mms_time - pd.Timedelta("5 second"), mms_time + pd.Timedelta("5 second")]
        mms_mec_trange = [mms_time - pd.Timedelta("30 second"),
                          mms_time + pd.Timedelta("30 second")]
        # Convert mms time range to string
        mms_trange = [mms_trange[0].strftime("%Y-%m-%d %H:%M:%S"),
                      mms_trange[1].strftime("%Y-%m-%d %H:%M:%S")]
        mms_mec_trange = [mms_mec_trange[0].strftime("%Y-%m-%d %H:%M:%S"),
                          mms_mec_trange[1].strftime("%Y-%m-%d %H:%M:%S")]
        # Convert mms time range to datetime
        mms_trange_time_object = [pd.to_datetime(mms_trange[0]),
                                  pd.to_datetime(mms_trange[1])]
        mms_mec_trange_time_object = [pd.to_datetime(mms_mec_trange[0]),
                                      pd.to_datetime(mms_mec_trange[1])]

        mms_mec_varnames = [f"mms{mms_probe_num}_mec_r_gsm"]
        _ = spd.mms.mec(trange=mms_mec_trange, varnames=mms_mec_varnames, probe=mms_probe_num,
                        data_rate="srvy", level="l2", time_clip=time_clip, latest_version=True)
        mms_mec_time = ptt.get_data("mms3_mec_r_gsm")[0]
        # Convert mms fgm time to datetime
        mms_mec_time = np.array([datetime.datetime.utcfromtimestamp(t) for t in mms_mec_time])

        # Position of MMS in GSM coordinates in earth radii (r_e) units
        r_e = 6378.137  # Earth radius in km
        mms_sc_pos = ptt.get_data("mms3_mec_r_gsm")[1:3][0] / r_e

        # TODO: Find out why adding "mms_fgm_varnames" as a variable causes the code to give out no
        # data.
        mms_fgm_varnames = [f"mms{mms_probe_num}_fgm_b_gsm_srvy_l2_bvec"]
        _ = spd.mms.fgm(trange=mms_trange, probe=mms_probe_num, time_clip=time_clip,
                        latest_version=True)
        # mms_fgm_time = ptt.get_data(mms_fgm_varnames[0])[0]

        mms_fgm_b_gsm = ptt.get_data(f"mms{mms_probe_num}_fgm_b_gsm_srvy_l2_bvec")[1:4][0]
        mms_fgm_time = ptt.get_data(f"mms{mms_probe_num}_fgm_b_gsm_srvy_l2_bvec")[0]
        # Convert mms fgm time to datetime
        mms_fgm_time = np.array([datetime.datetime.utcfromtimestamp(t) for t in mms_fgm_time])

        try:
            data_rate = "fast"
            mms_fpi_varnames = [f"mms{mms_probe_num}_dis_bulkv_gse_{data_rate}"]
            _ = spd.mms.fpi(trange=mms_trange, probe=mms_probe_num, data_rate=data_rate,
                            level="l2", datatype="dis-moms", varnames=mms_fpi_varnames,
                            time_clip=time_clip, latest_version=True)
            _ = spd.cotrans(name_in=f"mms{mms_probe_num}_dis_bulkv_gse_{data_rate}",
                            name_out=f"mms{mms_probe_num}_dis_bulkv_gsm_{data_rate}",
                            coord_in="gse", coord_out="gsm")
            mms_fpi_time = ptt.get_data(f"mms{mms_probe_num}_dis_bulkv_gsm_{data_rate}")[0]
            # Convert mms_fpi_time to datetime from unix time
            mms_fpi_time = np.array([datetime.datetime.utcfromtimestamp(x) for x in mms_fpi_time])
            mms_fpi_bulkv_gsm = ptt.get_data(f"mms{mms_probe_num}_dis_bulkv_gsm_{data_rate}")[1:4][0]
            if verbose:
                print(f"\n \033[1;31m {data_rate} mode data found for MMS{mms_probe_num} \033[0m \n")
        except:
            # Get the data from the FPI
            data_rate = "brst"
            mms_fpi_varnames = [f"mms{mms_probe_num}_dis_bulkv_gse_{data_rate}"]
            _ = spd.mms.fpi(trange=mms_trange, probe=mms_probe_num, data_rate=data_rate,
                            level="l2", datatype="dis-moms", varnames=mms_fpi_varnames,
                            time_clip=time_clip, latest_version=True)
            _ = spd.cotrans(name_in=f"mms{mms_probe_num}_dis_bulkv_gse_{data_rate}",
                            name_out=f"mms{mms_probe_num}_dis_bulkv_gsm_{data_rate}",
                            coord_in="gse", coord_out="gsm")
            mms_fpi_time = ptt.get_data(f"mms{mms_probe_num}_dis_bulkv_gsm_{data_rate}")[0]
            # Convert mms_fpi_time to datetime from unix time
            mms_fpi_time = [datetime.datetime.utcfromtimestamp(x) for x in mms_fpi_time]
            mms_fpi_bulkv_gsm = ptt.get_data(f"mms{mms_probe_num}_dis_bulkv_gsm_{data_rate}")[1:4][0]
            if verbose:
                print(f"\n \033[1;32m {data_rate} mode data found for MMS{mms_probe_num} \033[0m \n")
    else:
        mms_time = None
        mms_sc_pos = None
        mms_fgm_b_gsm = None
        mms_fpi_bulkv_gsm = None
        pass

    # Create the dataframe for OMNI data using omni_time_datetime as the index
    omni_df = pd.DataFrame({
        "time": omni_time,
        "bx_gsm": omni_bx_gse,
        "by_gsm": omni_by_gsm,
        "bz_gsm": omni_bz_gsm,
        "vx": omni_vx,
        "vy": omni_vy,
        "vz": omni_vz,
        "np": omni_np,
        "sym_h": omni_sym_h,
        "t_p": omni_t_p,
    }, index=omni_time_datetime)

    # Add UTC as time zone to the index of omni_df
    omni_df.index = omni_df.index.tz_localize("UTC")

    # Get the mean values of the parameters from OMNI data for the time range betwwen
    time_imf = np.nanmean(omni_df["time"].loc[omni_trange_time_object[0]:
                                              omni_trange_time_object[1]])
    b_imf_x = np.nanmean(omni_df["bx_gsm"].loc[omni_trange_time_object[0]:
                                               omni_trange_time_object[1]])
    b_imf_y = np.nanmean(omni_df["by_gsm"].loc[omni_trange_time_object[0]:
                                               omni_trange_time_object[1]])
    b_imf_z = np.nanmean(omni_df["bz_gsm"].loc[omni_trange_time_object[0]:
                                               omni_trange_time_object[1]])
    vx_imf = np.nanmean(omni_df["vx"].loc[omni_trange_time_object[0]:
                                          omni_trange_time_object[1]])
    vy_imf = np.nanmean(omni_df["vy"].loc[omni_trange_time_object[0]:
                                          omni_trange_time_object[1]])
    vz_imf = np.nanmean(omni_df["vz"].loc[omni_trange_time_object[0]:
                                          omni_trange_time_object[1]])
    np_imf = np.nanmean(omni_df["np"].loc[omni_trange_time_object[0]:
                                          omni_trange_time_object[1]])
    sym_h_imf = np.nanmean(omni_df["sym_h"].loc[omni_trange_time_object[0]:
                                                omni_trange_time_object[1]])
    tp_imf = np.nanmean(omni_df["t_p"].loc[omni_trange_time_object[0]:
                                           omni_trange_time_object[1]])

    if (b_imf_z > 15 or b_imf_z < -18):
        warnings.warn(
            f"The given parameters produced the z-component of IMF field (b_imf_z) {b_imf_z} nT,"
            f"which is out of range in which model is valid (-18 nT < b_imf_z < 15 nT)"
        )

    time_imf_hrf = datetime.datetime.utcfromtimestamp(time_imf)

    v_imf = [vx_imf, vy_imf, vz_imf]
    b_imf = [b_imf_x, b_imf_y, b_imf_z]

    imf_clock_angle = np.arctan2(b_imf[1], b_imf[2]) * 180 / np.pi
    if imf_clock_angle < 0:
        imf_clock_angle += 360
    if mms_probe_num is not None:
        mean_mms_sc_pos = np.round(np.nanmean(mms_sc_pos, axis=0), decimals=2)
    else:
        mean_mms_sc_pos = None

    print("IMF parameters found:")
    if (verbose):
        print(tabulate(
            [["Time of observation (UTC)", f"{time_imf_hrf}"],
             ["IMF Magnetic field [GSM] (nT)", f"[{b_imf[0]:.2f}, {b_imf[1]:.2f}, {b_imf[2]:.2f}]"],
             ["IMF Proton density (1/cm^-3)", f"{np_imf:.2f}"],
             ["IMF Plasma velocity (km/sec)", f"[{v_imf[0]:.2f}, {v_imf[1]:.2f}, {v_imf[2]:.2f}]"],
             ["IMF clock angle (degrees)", f"{imf_clock_angle:.2f}"],
             ["IMF Sym H", f"{sym_h_imf:.2f}"],
             ["MMS position (GSM) (R_E)", f"[{mean_mms_sc_pos[0]:.2f}, {mean_mms_sc_pos[1]:.2f}, "
                                          f"{mean_mms_sc_pos[2]:.2f}]"]],
            headers=["Parameter", "Value"], tablefmt="fancy_grid", floatfmt=".2f",
            numalign="center"))

    # Check if the values are finite, if not then assign a default value to each of them
    if ~(np.isfinite(np_imf)):
        np_imf = 5
    if ~(np.isfinite(vx_imf)):
        vx_imf = -500
    if ~(np.isfinite(vy_imf)):
        vy_imf = 0
    if ~(np.isfinite(vz_imf)):
        vz_imf = 0
    if ~(np.isfinite(sym_h_imf)):
        sym_h_imf = -1

    m_proton = 1.672e-27  # Mass of proton in SI unit

    rho = np_imf * m_proton * 1.15  # NOTE to self: Unit is fine, do not worry about it

    #  Solar wind ram pressure in nPa, including roughly 4% Helium++ contribution
    p_dyn = 1.6726e-6 * 1.15 * np_imf * (vx_imf**2 + vy_imf**2 + vz_imf**2)

    if (p_dyn > 8.5 or p_dyn < 0.5):
        warnings.warn(
            f"The given parameters produced a dynamic pressure of {p_dyn} nPa which is out of"
            f" range in which model is valid (0.5 nPa < p_dyn < 8.5 nPa)",
        )
    param = [p_dyn, sym_h_imf, b_imf_y, b_imf_z, 0, 0, 0, 0, 0, 0]

    # Compute the dipole tilt angle
    ps = gp.recalc(time_imf)

    # Create a dataframe for mms fgm data
    df_fgm = pd.DataFrame(data=mms_fgm_b_gsm, columns=["Bx", "By", "Bz"], index=mms_fgm_time)

    # Get the mean of the fgm data for time range between mms_trange_time_object[0] and
    bx_mean = np.nanmean(df_fgm["Bx"].loc[mms_trange_time_object[0]:mms_trange_time_object[1]])
    by_mean = np.nanmean(df_fgm["By"].loc[mms_trange_time_object[0]:mms_trange_time_object[1]])
    bz_mean = np.nanmean(df_fgm["Bz"].loc[mms_trange_time_object[0]:mms_trange_time_object[1]])

    # Create a dataframe for mms fpi data
    df_fpi = pd.DataFrame(data=mms_fpi_bulkv_gsm, columns=["Vx", "Vy", "Vz"], index=mms_fpi_time)

    # Get the mean of the fpi data for time range between mms_trange_time_object[0] and
    vx_mean = np.nanmean(df_fpi["Vx"].loc[mms_trange_time_object[0]:mms_trange_time_object[1]])
    vy_mean = np.nanmean(df_fpi["Vy"].loc[mms_trange_time_object[0]:mms_trange_time_object[1]])
    vz_mean = np.nanmean(df_fpi["Vz"].loc[mms_trange_time_object[0]:mms_trange_time_object[1]])

    # Create a dataframe for mms sc position data
    df_mec = pd.DataFrame(data=mms_sc_pos, columns=["X", "Y", "Z"], index=mms_mec_time)

    # Get the mean of the sc position data for time range between mms_trange_time_object[0] and
    x_mean = np.nanmean(df_mec["X"].loc[
                                    mms_mec_trange_time_object[0]:mms_mec_trange_time_object[1]])
    y_mean = np.nanmean(df_mec["Y"].loc[
                                    mms_mec_trange_time_object[0]:mms_mec_trange_time_object[1]])
    z_mean = np.nanmean(df_mec["Z"].loc[
                                    mms_mec_trange_time_object[0]:mms_mec_trange_time_object[1]])

    # Make a dictionary of all the solar wind parameters
    sw_dict = {}
    sw_dict["time"] = time_imf
    sw_dict["b_imf"] = b_imf
    sw_dict["rho"] = rho
    sw_dict["ps"] = ps
    sw_dict["p_dyn"] = p_dyn
    sw_dict["sym_h"] = sym_h_imf
    sw_dict["t_p"] = tp_imf
    sw_dict["imf_clock_angle"] = imf_clock_angle
    sw_dict["param"] = param
    sw_dict["mms_time"] = mms_time
    sw_dict["mms_sc_pos"] = [x_mean, y_mean, z_mean]
    sw_dict["mms_b_gsm"] = [bx_mean, by_mean, bz_mean]
    sw_dict["mms_v_gsm"] = [vx_mean, vy_mean, vz_mean]

    return sw_dict


def rx_model(
    probe=None,
    trange=["2016-12-24 15:08:00", "2016-12-24 15:12:00"],
    dt=5,
    omni_level="hro",
    mms_probe_num="3",
    model_type="t96",
    m_p=0.5,
    dr=0.5,
    min_max_val=15,
    y_min=None,
    y_max=None,
    z_min=None,
    z_max=None,
    save_data=False,
    nprocesses=None
):
    """
    This function computes the magnetosheath and magnetospheric magnetic fields using the T96 model
    and  solar wind paramters and returns the value of those fields as well as the value of other
    parameters associated with dayside reconnection, such as shear angle, reconnection energy,
    exhaust velocity, and the bisection field based on bisection model, for both the magnetospheric
    and magnetosheath bisection fields.

    Parameters
    ----------
    probe : str
        The probe to be used for the computation.
        Options: "mms1", "mms2", "mms3", "mms4"
    trange : list or an array of length 2
        The time range to use. Should in the format [start, end], where start and end times should
        be a string in the format "YYYY-MM-DD HH:MM:SS".
    dt : float
        The time interval to use in case the "trange" has only one element.
    omni_level : str, optional
        The omni-data level to be used for the computation. Default is "hro".
        Options: "hro", "hro1"
    mms_probe_num : str. optional
        The MMS probe number to be used for the computation. Default is "3".
        Options: "1", "2", "3", "4"
    model_type : str
        The model type to be used for the computation. Default is "t96".
        Options: "t96", "t01"
    m_p : float
        Thickness of the magnetosphere in earth radius units . Default is 0.5 R_E.
    dr : float
        Finess of the grid for model computation. Default is 0.5 R_E.
    min_max_val : float
        The minimum and maximum values of the y and z-axis to be used for the computation. Default
        is 15, meaning the model will be computed for -15 < y < 15 and -15 < z < 15.
    y_min : float, optional
        The minimum value of the y-axis to be used for the computation. Default is None, in which
        case y_min is set to -min_max_val.
    y_max : float, optional
        The maximum value of the y-axis to be used for the computation. Default is None, in which
        case y_max is set to min_max_val.
    z_min : float, optional
        The minimum value of the z-axis to be used for the computation. Default is None, in which
        case z_min is set to -min_max_val.
    z_max : float, optional
        The maximum value of the z-axis to be used for the computation. Default is None, in which
        case z_max is set to min_max_val.
    save_data : bool, optional
        Whether to save the data or not. Default is False. If True, the data will be saved in a
        "HDF5" file.
    nprocesses : int, optional
        The number of processes to use for the computation. Default is None, in which case the
        number of processes will be set to the number of cores in the system.
    """

    if y_min is None:
        y_min = -min_max_val
    if y_max is None:
        y_max = min_max_val
    if z_min is None:
        z_min = -min_max_val
    if z_max is None:
        z_max = min_max_val

    # If trange has only one element, then make it a list of length 2 with "dt" minutes of padding
    if len(trange) == 1:
        # Check if trange is a datetime object
        if isinstance(trange[0], datetime.datetime):
            trange_date = trange[0]
        else:
            trange_date = datetime.datetime.strptime(trange[0], "%Y-%m-%d %H:%M:%S")
        trange_date_min = trange_date - datetime.timedelta(minutes=dt)
        trange_date_max = trange_date + datetime.timedelta(minutes=dt)
        trange = [trange_date_min.strftime("%Y-%m-%d %H:%M:%S"),
                  trange_date_max.strftime("%Y-%m-%d %H:%M:%S")]
        # Add timezones to trange (UTC)
        trange = [trange[0] + "Z", trange[1] + "Z"]


    # Get the solar wind parameters for the model
    sw_params = get_sw_params(probe=probe, omni_level=omni_level, trange=trange,
                              mms_probe_num=mms_probe_num, verbose=True)
    print(sw_params)

    n_arr_y = int((y_max - y_min) / dr) + 1
    n_arr_z = int((z_max - z_min) / dr) + 1
    bx = np.full((n_arr_y, n_arr_z), np.nan)
    by = np.full((n_arr_y, n_arr_z), np.nan)
    bz = np.full((n_arr_y, n_arr_z), np.nan)
    b_msx = np.full((n_arr_y, n_arr_z), np.nan)
    b_msy = np.full((n_arr_y, n_arr_z), np.nan)
    b_msz = np.full((n_arr_y, n_arr_z), np.nan)
    shear = np.full((n_arr_y, n_arr_z), np.nan)
    rx_en = np.full((n_arr_y, n_arr_z), np.nan)
    va_cs = np.full((n_arr_y, n_arr_z), np.nan)
    bisec_msp = np.full((n_arr_y, n_arr_z), np.nan)
    bisec_msh = np.full((n_arr_y, n_arr_z), np.nan)

    x_shu = np.full((n_arr_y, n_arr_z), np.nan)
    y_shu = np.full((n_arr_y, n_arr_z), np.nan)
    z_shu = np.full((n_arr_y, n_arr_z), np.nan)

    b_msx = np.full((n_arr_y, n_arr_z), np.nan)
    b_msy = np.full((n_arr_y, n_arr_z), np.nan)
    b_msz = np.full((n_arr_y, n_arr_z), np.nan)

    # Shue et al.,1998, equation 10
    ro = (10.22 + 1.29 * np.tanh(0.184 * (sw_params["b_imf"][2] + 8.14))) * (
        sw_params["p_dyn"])**(-1.0 / 6.6)

    # Shue et al.,1998, equation 11
    alpha = (0.58 - 0.007 * sw_params["b_imf"][2]) * (1 + 0.024 * np.log(sw_params["p_dyn"]))
    rmp = ro * (2 / (1 + np.cos(0.0))) ** alpha  # Stand off position of the magnetopause

    len_y = int((y_max - y_min) / dr) + 1
    len_z = int((z_max - z_min) / dr) + 1

    if nprocesses is None:
        p = mp.Pool()
    else:
        p = mp.Pool(processes=nprocesses)

    input = ((j, k, y_max, z_max, dr, m_p, ro, alpha, rmp, sw_params, model_type)
             for j in range(len_y) for k in range(len_z))

    print("Running the model \n")
    res = p.map(model_run, input)
    print("Model run complete \n")

    p.close()
    p.join()

    for r in res:
        j = r[0]
        k = r[1]
        bx[j, k] = r[2]
        by[j, k] = r[3]
        bz[j, k] = r[4]

        shear[j, k] = r[5]
        rx_en[j, k] = r[6]
        va_cs[j, k] = r[7]
        bisec_msp[j, k] = r[8]
        bisec_msh[j, k] = r[9]

        x_shu[j, k] = r[10]
        y_shu[j, k] = r[11]
        z_shu[j, k] = r[12]

        b_msx[j, k] = r[13]
        b_msy[j, k] = r[14]
        b_msz[j, k] = r[15]

    if save_data:
        try:
            today_date = datetime.datetime.today().strftime("%Y-%m-%d")
            fn = f"../data/all_data_rx_model_{dr}re_{m_p}mp_{model_type}_{today_date}.h5"
            data_file = hf.File(fn, "w")

            data_file.create_dataset("bx", data=bx)
            data_file.create_dataset("by", data=by)
            data_file.create_dataset("bz", data=bz)

            data_file.create_dataset("b_msx", data=b_msx)
            data_file.create_dataset("b_msy", data=b_msy)
            data_file.create_dataset("b_msz", data=b_msz)

            data_file.create_dataset("shear", data=shear)
            data_file.create_dataset("rx_en", data=rx_en)
            data_file.create_dataset("va_cs", data=va_cs)
            data_file.create_dataset("bisec_msp", data=bisec_msp)
            data_file.create_dataset("bisec_msh", data=bisec_msh)

            data_file.create_dataset("x_shu", data=x_shu)
            data_file.create_dataset("y_shu", data=y_shu)
            data_file.create_dataset("z_shu", data=z_shu)

            data_file.close()
            print(f"Date saved to file {fn} \n")
        except Exception as e:
            print(e)
            print(f"Data not saved to file {fn}. Please make sure that file name is correctly" +
                  " assigned and that the directory exists and you have write permissions")

    return (bx, by, bz, shear, rx_en, va_cs, bisec_msp, bisec_msh, sw_params, x_shu, y_shu, z_shu,
            b_msx, b_msy, b_msz)


def line_fnc(
    r0=np.array([0, 0, 0]),
    b_msh=np.array([0, 0, 0]),
    r=0,
):
    """
    Function to compute the equation of a line along the direction of the magnetic field.

    Parameters
    ----------
    r0 : array
        Starting position of the line
    b_msh : array, optional
        Magnetic field from magnetosheath at the magnetopause
    r : float, optional
        Position of the point where the line is computed
    """
    # Compute the direction of the magnetic field line
    b_msh_norm = np.linalg.norm(b_msh)
    b_msh_dir = b_msh / b_msh_norm

    # return the line function
    return r0 + r * b_msh_dir


def line_fnc_der(x, y):
    """
    Function to give a line interpolation function

    Parameters
    ----------
    x : array
        x coordinates of the points
    y : array
        y coordinates of the points

    Returns
    -------
    line_intrp : function
        Function to interpolate the line
    """

    nans, x_nan = nan_helper(x)
    x[nans] = np.interp(x_nan(nans), x_nan(~nans), x[~nans])

    nans, y_nan = nan_helper(y)
    y[nans] = np.interp(y_nan(nans), y_nan(~nans), y[~nans])

    line_intrp = sp.interpolate.CubicSpline(x, y)
    return line_intrp


def target_fnc(r, r0, b_msh, line_fnc, line_intrp):
    p_line = line_fnc(r0=r0, b_msh=b_msh, r=r)
    z_surface = line_intrp(p_line[1])
    return np.sum((p_line[2] - z_surface)**2)


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to "equivalent" indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]
