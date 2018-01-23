#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     fit_1D_poly_nestle.py                                             #
#                                                                             #
# PURPOSE:  Example of using Nestle module to fit a polynomial to some data   #
#                                                                             #
# MODIFIED: 23-Jan-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

# Input dataset 
specDat = "polySpec.dat"

# Prior bounds of parameters in 3rd order polynomial model
# y = p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3
boundsLst = [[  0.0,   2.0],    #   0 < p[0] < 2
             [ -1.0,   1.0],    #  -1 < p[1] < 1 
             [ -1.0,   1.0],    #  -1 < p[2] < 1 
             [ -1.0,   1.0]]    #  -1 < p[4] < 1

# Prior type ("uniform" or "normal")
priorType = "uniform"


#=============================================================================#
import os
import sys
import numpy as np
import matplotlib as mpl
import pylab as pl
from scipy.special import ndtri

from Imports import nestle
from Imports import corner


#-----------------------------------------------------------------------------#
def main():
    
    # Read in the spectrum
    specArr = np.loadtxt(specDat, dtype="float64", unpack=True)
    xArr = specArr[0] / 1e9   # GHz -> Hz for this dataset 
    yArr = specArr[1] 
    dyArr = specArr[4]

    # Must define the lnlike() here so data can be inserted
    def lnlike(p):
        return -0.5*(np.sum( (yArr-model(p)(xArr))**2/dyArr**2 ))
    
    # Set the prior function given the bounds
    priorTr = prior(boundsLst, priorType)
    nDim = len(boundsLst)
    
    # Run nested sampling
    res = nestle.sample(loglikelihood   = lnlike,
                        prior_transform = priorTr,
                        ndim            = nDim,
                        npoints         = 300,
                        method          = "single")
    
    # Weighted average and covariance:
    p, cov = nestle.mean_and_cov(res.samples, res.weights)
    
    # Summary of run
    print("-"*80)
    print("NESTLE SUMMARY:")
    print(res.summary())
    print("")
    print("-"*80)
    print("RESULTS:")
    for i in range(len(p)):
        print("p%d = %.4f +/- %.4f" % (i, p[i], np.sqrt(cov[i, i])))
    
    # Plot the data and best fit
    plot_model(p, xArr, yArr, dyArr)

    # Plot the triangle plot
    fig = corner.corner(res.samples,
                        weights = res.weights,
                        labels  = ["p" + str(i) for i in range(nDim)],
                        range   = [0.99999]*nDim,
                        truths  = p,
                        bins    = 30)
    fig.show()
    print("Press <Return> to finish:")
    raw_input()

    
#-----------------------------------------------------------------------------#
def model(p):
    """ Returns a function to evaluate the model """
    
    def rfunc(x):
        y = p[0] + p[1]*x + p[2]*x**2. + p[3]*x**3.
        return y
    
    return rfunc


#-----------------------------------------------------------------------------#
def prior(boundsLst, priorType="uniform"):
    """ Returns a function to transform (0-1) range to the distribution of 
    values for each parameter """

    b = np.array(boundsLst, dtype="f4")
    r = b[:,1]-b[:,0]
    sigma = r/2.0
    mu = b[:,0] + sigma
    
    if priorType == "normal":
        def rfunc(p):
            return mu + sigma * ndtri(p)

    else:
        def rfunc(p):
            return b[:,0] + p * r
        
    return rfunc


#-----------------------------------------------------------------------------#
def plot_model(p, x, y, dy, scaleX=1.0):

    # Make the model curve
    nSamples = 100
    dXSamp = (np.max(x) - np.min(x)) / nSamples
    iLst = np.arange(nSamples, dtype='float32')
    xSamp = np.min(x) + dXSamp * iLst
    ySamp = model(p)(xSamp)

    # Plot the channels and fit
    fig = pl.figure()
    fig.set_size_inches([8,4])
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.plot(xSamp*scaleX, ySamp, color='b',marker='None',mfc='w',
            mec='g', ms=10, label='none', lw=1.0)
    ax.errorbar(x=x*scaleX , y=y, yerr=dy, mfc='none', ms=4, fmt='D',
                ecolor='red', elinewidth=1.0, capsize=2)
    fig.show()

    
#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
