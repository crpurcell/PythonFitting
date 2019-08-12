#!/usr/bin/env python


#specIdat = '034205.4-370322.00_specI.dat'
#specIdat = 'Source8.dat'
specIdat = 'HotSpot.dat'
order = 5

#=============================================================================#
import os, sys, shutil
import math as m
import numpy as np
from mpfit import mpfit
import pylab as pl
import matplotlib as mpl
from scipy import nanmedian
from scipy import nanmean


#-----------------------------------------------------------------------------#
def main():

    # Read in the spectrum
    specIArr = np.loadtxt(specIdat, dtype="float64", unpack=True)
    specIArr[0] /= 1e9 # Hz -> GHz
 
    # Autoscale the data
    scaleX = 1.0 #np.nanmax(specIArr[0])
    scaleY = 1.0 #np.nanmax(specIArr[1])
    specIArr[0] /= scaleX
    specIArr[1] /= scaleY
    specIArr[4] /= scaleY
    
     
    # Fit the spectrum
    mp = fit_spec_poly5(specIArr[0], specIArr[1], specIArr[4], order)
    print("STATUS:", mp.status)
    print("CHISQ:", mp.fnorm)
    print("CHISQred:", mp.fnorm/(len(specIArr[0])-order-1))
    print("NITER:", mp.niter)
    print(" P:", mp.params*scaleY)
    print("dP:", mp.perror*scaleY)
    print("scaleX, scaleY:", scaleX, scaleY)
    
    # Plot the model spectrum
    plot_spec_poly5(mp.params*scaleY, specIArr[0], specIArr[1]*scaleY,
                    specIArr[4]*scaleY, scaleX)


#-----------------------------------------------------------------------------#
def fit_spec_poly5(xData, yData, dyData, order=5):
    """
    Fit a 5th order polynomial to a spectrum. To avoid overflow errors the
    X-axis data should not be large numbers (e.g.: x10^9 Hz; use GHz instead).
    """

    # Lower order limit is a line with slope
    if order<1:
        order = 1
    if order>5:
        order = 5
        
    # Estimate starting coefficients
    C1 = nanmean(np.diff(yData)) / nanmedian(np.diff(xData))
    ind = int(np.median(np.where(~np.isnan(yData))))
    C0 = yData[ind] - (C1 * xData[ind])
    C5 = 0.0
    C4 = 0.0
    C3 = 0.0
    C2 = 0.0
    inParms=[ {'value': C5, 'parname': 'C5'},
              {'value': C4, 'parname': 'C4'},
              {'value': C3, 'parname': 'C3'},
              {'value': C2, 'parname': 'C2'},
              {'value': C1, 'parname': 'C1'},
              {'value': C0, 'parname': 'C0'} ]
    
    # Set the polynomial order
    for i in range(len(inParms)):
        if len(inParms)-i-1>order:
            inParms[i]['fixed'] = True
        else:
            inParms[i]['fixed'] = False

    # Function to evaluate the difference between the model and data.
    # This is minimised in the least-squared sense by the fitter
    def errFn(p, fjac=None):
        status = 0
        return status, (poly5(p)(xData) - yData)/dyData

    # Use mpfit to perform the fitting
    mp = mpfit(errFn, parinfo=inParms, quiet=True)
    return mp
    

#-----------------------------------------------------------------------------#
def plot_spec_poly5(p, x, y, dy, scaleX=1.0):

    # Make the model curve
    nSamples = 100
    dXSamp = (np.max(x) - np.min(x)) / nSamples
    iLst = np.arange(nSamples, dtype='float32')
    xSamp = np.min(x) + dXSamp * iLst
    ySamp = poly5(p)(xSamp)

    # Plot the channels and fit
    fig = pl.figure()
    fig.set_size_inches([8,4])
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Amplitude (mJy)')
    ax.plot(xSamp*scaleX, ySamp, color='b',marker='None',mfc='w',
            mec='g', ms=10, label='none', lw=1.0)
    ax.errorbar(x=x*scaleX , y=y, yerr=dy, mfc='none', ms=4, fmt='D',
                ecolor='red', elinewidth=1.0, capsize=2)
    fig.show()
    print("Press <Return> to finish:",)
    input()

    
    
#-----------------------------------------------------------------------------
def poly5(p):
    """
    Function which returns another function to evaluate a polynomial.
    The subfunction can be accessed via 'argument unpacking' like so:
    'y = poly5(p)(*x)', where x is a vector of X values and p is a
    vector of polynomial coefficients.
    """

    # Fill out the vector to length 6
    p = np.append(np.zeros((6-len(p))), p)
    
    def rfunc(x):
        y = p[0]*x**5.0 + p[1]*x**4.0 + p[2]*x**3.0 + p[3]*x**2.0 + p[4]*x +p[5]
        return y
    return rfunc


#-----------------------------------------------------------------------------#
main()
