#!/usr/bin/env python
"""
Fit a 2D Gaussian in gridded image data, rather than sampled data.
This version uses the MPFIT routines.
"""

import os
import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpfit import mpfit


#-----------------------------------------------------------------------------#
def twodgaussian(params, shape=None):
    """
    Function to generate an image containing a Gaussian.

    If called without a shape, returns a function with the parameters
    'baked in' that can be used by a fitter. If called with a shape, it
    evaluates the function and returns a Numpy array.
    """

    amp, x0, y0, sig_x, sig_y, pa = params
    pa = np.radians(pa)

    def gauss(y, x):
        st = np.sin(pa)**2
        ct = np.cos(pa)**2
        s2t = np.sin(2*pa)
        a = (ct/sig_x**2 + st/sig_y**2)/2
        b = s2t/4 *(1/sig_y**2-1/sig_x**2)
        c = (st/sig_x**2 + ct/sig_y**2)/2
        v = amp * np.exp(-1*(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
        return v

    if shape is not None:
        return gauss(*np.indices(shape))
    else:
        return gauss


#-----------------------------------------------------------------------------#
def moments(data):
    """
    Calculate the moments of 2D data.

    Returns: height, x, y, width_x, width_y, pa
    """

    total = data.sum()
    height = data.max()
    xi, yi = np.indices(data.shape)
    x = (xi * data).sum() / total
    y = (yi * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    pa = 0.0

    return height, x, y, width_x, width_y, pa


#-----------------------------------------------------------------------------#
def vect_to_mpfit_parms(p, p_names=None, p_fixed=None, p_limits=None):
    """
    Convert a vector of parameter values to a MPFIT parameter structure.
    The structure is a list of dictionaries with this format:

      parms=[ {'value': 5.1,
               'fixed': False,
               'parname': 'amp',
               'limited': [False, False]}, ... ]

    The 'fixed' keyword freezes the variable at the provided value,
    the 'parname'keyword allows the provision of a name (including
    LaTex code), and the 'limited' keyword supports the setting of
    fitting bounds.

    """

    mpfit_parm_lst = []
    for idx, value in enumerate(p):
        mpfit_parm_lst.append({'value': value,
                               'fixed': False,
                               'parname': f"Var_{idx}",
                               'limited': [False, False]})
    if p_names is not None:
        if len(p_names) == len(p):
            for idx, value in enumerate(p_names):
                mpfit_parm_lst[idx]["parname"] = value
    if p_fixed is not None:
        if len(p_fixed) == len(p):
            for idx, value in enumerate(p_fixed):
                mpfit_parm_lst[idx]["fixed"] = bool(value)
    if p_limits is not None:
        if len(p_limits) == len(p):
            for idx, value in enumerate(p_names):
                mpfit_parm_lst[idx]["limited"] = value

    return mpfit_parm_lst


#-----------------------------------------------------------------------------#
def get_parm_vector(parms, field_name="value"):
    """
    Get a vector of parameters given a field name.
    Allowed field names are ['value', 'fixed', 'parname', 'limited'].
    """

    if not field_name in ['value', 'fixed', 'parname', 'limited']:
        return [None] * len(parms)

    val_lst = []
    for idx, e in enumerate(parms):
        val_lst.append(e[field_name])

    return val_lst


#-----------------------------------------------------------------------------#
if __name__ == '__main__':


    # Generate a noisy image containing a gaussian
    #      [amp, x0,    y0,    sig_x, sig_y, pa]
    p_in = [1.0, 100.0, 100.0, 10.0, 30.0, 20.0]
    shape = (200, 200)
    data_arr = twodgaussian(p_in, shape)
    noise_frac = 0.8
    data_arr += (np.random.random(data_arr.shape) - 0.5) * noise_frac

    # Estimate the initial parameters by calculating moments
    p = moments(data_arr)

    # Convert to a MPFIT parameter vector
    p_names = ["amp", "x0", "y0", "sig_x", "sig_y", "pa"]
    inparms = vect_to_mpfit_parms(p, p_names)

    # Fit the Gaussian using MPFIT
    def err_mpfit_fn(p, fjac=None):
        status = 0
        return status, np.ravel(twodgaussian(p)(*np.indices(data_arr.shape))
                                - data_arr)
    mp = mpfit(err_mpfit_fn, parinfo=inparms, quiet=False)
    p = mp.params

    # Calculate goodness-of-fit parameters
    n_free_parms = sum(~np.array(get_parm_vector(inparms, "fixed")))
    dof = len(data_arr) - n_free_parms - 1
    chi_sq = mp.fnorm
    chi_sq_red = mp.fnorm / dof
    n_iter = mp.niter
    p_err = mp.perror

    # Feedback to user
    print("="*80)
    print("[INFO] Final fit values:\n")
    for i_ in range(len(inparms)):
        print("\t%s = %f +/- %f" % (inparms[i_]['parname'],
                                    mp.params[i_],
                                    mp.perror[i_]))
    print("\n[INFO] Goodness-of-fit metrics:\n")
    print("\tDoF:           %d" % dof)
    print("\tChiSq:         %.1f" % chi_sq)
    print("\tChiSq Reduced: %.1f" % chi_sq_red)
    print("\tN Iter:        %d\n" % n_iter)

    # Plot the data and fit ellipse
    print("[INFO] Plotting fit ")
    fig = plt.figure(figsize=(18,4.3))
    ax1 = fig.add_subplot(1,3,1)
    cax1 = ax1.imshow(data_arr, origin='lower', cmap=mpl.cm.jet)
    cbar1=fig.colorbar(cax1, pad=0.0)
    sigma2fwhm = math.sqrt(8*math.log(2))
    ellipse = Ellipse(
        xy=(p[1], p[2]),
        width=p[3] * sigma2fwhm,
        height=p[4]* sigma2fwhm,
        angle=-1*p[5],
        edgecolor="magenta",
        fc="None",
        lw=2)
    ax1.add_patch(ellipse)
    ax1.set_title("Generated Data")
    ax1.set_xlim(0, shape[-1]-1)
    ax1.set_ylim(0, shape[-2]-1)
    ax1.set_aspect('equal')

    # Plot the model fit
    data_fit_arr = twodgaussian(p, shape)
    ax2 = fig.add_subplot(1,3,2)
    cax2 = ax2.imshow(data_fit_arr, origin='lower', cmap=mpl.cm.jet)
    cbar2=fig.colorbar(cax2, pad=0.0)
    ax2.set_title("Model Fit")

    # Plot the residual
    ax3 = fig.add_subplot(1,3,3)
    data_res_arr = data_arr - data_fit_arr
    cax3 = ax3.imshow(data_res_arr, origin='lower', cmap=mpl.cm.jet)
    cbar2=fig.colorbar(cax3, pad=0.0)
    ax3.set_title("Residual")

    fig.canvas.draw()
    fig.show()
    input("Press <RETURN> to continue ...")
