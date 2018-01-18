#!/usr/bin/env python


rmsfDat = '034205.4-370322.00_RMSF.dat'
#rmsfDat = 'test.dat'

#=============================================================================#
import os, sys, shutil
import math as m
import numpy as np
import numpy.ma as ma
from mpfit import mpfit
import pylab as pl
import matplotlib as mpl
from scipy.stats.stats import nanmedian
from scipy.stats.stats import nanmean


#-----------------------------------------------------------------------------#
def main():

    # Read in the spectrum
    RMSF = np.loadtxt(rmsfDat, dtype="float64", delimiter=' ', unpack=True)
    phi = RMSF[0]
    absRMSF = np.sqrt(RMSF[1]**2.0 + RMSF[2]**2.0)
    
    # Fit the spectrum
    p, success = fit_rmsf(phi, absRMSF)
    print p, success
    # Plot the model spectrum
    plot_rmsf_gauss(p, phi, absRMSF)


#-----------------------------------------------------------------------------#
def fit_rmsf(xData, yData, thresh=0.3):
    """
    Fit the main lobe of the RMSF with a Gaussian function. Sidelobes beyond
    a threshold near the first null are masked out.
    """

    # Detect the peak and mask off the sidelobes
    msk = detect_peak(yData, thresh)
    validIndx = np.where(msk==1.0)
    xData = xData[validIndx]
    yData = yData[validIndx]
    
    # Estimate starting parameters
    a = 1.0
    b = xData[np.argmax(yData)]
    w = np.nanmax(xData)-np.nanmin(xData)
    
    # Estimate starting parameters
    inParms=[ {'value': a, 'fixed':False, 'parname': 'amp'},
              {'value': b, 'fixed':False, 'parname': 'offset'},
              {'value': w, 'fixed':False, 'parname': 'width'}]

    # Function which returns another function to evaluate a Gaussian
    def gauss1D(p):
        a, b, w = p
        gfactor = 2.0 * m.sqrt(2.0 * m.log(2.0))
        s = w / gfactor    
        def rfunc(x):
            y = a * np.exp(-(x-b)**2.0 /(2.0 * s**2.0))
            return y
        return rfunc
    
    # Function to evaluate the difference between the model and data.
    # This is minimised in the least-squared sense by the fitter
    def errFn(p, fjac=None):
        status = 0
        return status, gauss1D(p)(xData) - yData
    
    # Use mpfit to perform the fitting
    mp = mpfit(errFn, parinfo=inParms, quiet=True)
    coeffs = mp.params

    return mp.params, mp.status

    
#-----------------------------------------------------------------------------#
def detect_peak_old(a, slope=0.0):
    """
    Detect the extent of the peak in the array by looking for where the slope
    turns over.  The highest peak is detected and data followed until the
    first null. Assumes that the peak is well sampled and can only have two
    'highest' channels.
    """
    iPk = np.argmax(a)
    d = np.diff(a)
    off = 0
    if d[iPk]>=0.0:
        off = 1
    dl = np.flipud(d[:iPk])    
    dr = d[off+iPk:]
    iL = iPk - np.min(np.where(dl<=slope)) + 1
    iR = iPk + off + np.min(np.where(dr>=slope))
    msk = np.zeros_like(a)
    msk[iL:iR] = 1
    
    return msk
    
#-----------------------------------------------------------------------------#
def detect_peak(a, thresh=0.3):
    """
    Detect the extent of the peak in the array by looking for where the slope
    changes to flat. The highest peak is detected and data and followed until
    the slope flattens to a threshold.
    """
    
    iPk = np.argmax(a)
    d = np.diff(a)
    g1 = np.gradient(a)
    g2 = np.gradient(g1)
    
    threshPos = np.nanmax(d) * thresh
    threshNeg = -1 * threshPos

    # Start searching away from the peak zone
    g2L = np.flipud(g2[:iPk])
    g2R = g2[iPk+1:]
    iL = iPk - np.min(np.where(g2L>=0))
    iR = iPk + np.min(np.where(g2R>=0)) + 1
    g1[iL:iR] = np.nan
    
    # Search for the threshold crossing point
    g1L = np.flipud(g1[:iPk])
    g1R = g1[iPk+1:]
    iL = iPk - np.min(np.where(g1L<=threshPos))
    iR = iPk + np.min(np.where(g1R>=threshNeg))
    msk = np.zeros_like(a)
    msk[iL:iR] = 1

    # DEBUG
    if False:
        pl.plot(a, marker='.')
        pl.plot(g1, marker='^')
        pl.plot(msk, marker='o')
        pl.plot(np.ones_like(a)*threshPos, color='k')
        pl.plot(np.ones_like(a)*threshNeg, color='k')        
        pl.axhline(0, color='grey')
        pl.show()
    
    return msk
    

#-----------------------------------------------------------------------------
def gauss(p):

    a, b, w = p
    gfactor = 2.0 * m.sqrt(2.0 * m.log(2.0))
    s = w / gfactor
    
    def rfunc(x):
        y = a * np.exp(-(x-b)**2.0 /(2.0 * s**2.0))
        return y
    return rfunc

#-----------------------------------------------------------------------------#
def plot_rmsf_gauss(p, x, y):

    # Make the model curve
    nSamples = 1000
    dXSamp = (np.max(x) - np.min(x)) / nSamples
    iLst = np.arange(nSamples, dtype='float32')
    xSamp = np.min(x) + dXSamp * iLst
    ySamp = gauss(p)(xSamp)
    
    # Plot the channels and fit
    fig = pl.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.plot(x, y, color='g',marker='.',mfc='b',
            mec='b', ms=5, label='none', lw=1.0)
    ax.plot(xSamp, ySamp, color='k',marker='None',mfc='w',
            mec='g', ms=10, label='none', lw=1.0)
    pl.show()

    

#-----------------------------------------------------------------------------#
main()
