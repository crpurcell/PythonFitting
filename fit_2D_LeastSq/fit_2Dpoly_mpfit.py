#!/usr/bin/env python

# Initial model parameters
inParms=[ {'value': 5.1,
           'fixed': False,
           'parname': 'amp',
           'limited': [False, False]},

          {'value': 1.0,
           'fixed': False,
           'parname': 'x1',
           'limited': [False, False]},

          {'value': 1.0,
           'fixed': False,
           'parname': 'x2',
           'limited': [False, False]},

          {'value': 1.0,
           'fixed': False,
           'parname': 'x3',
           'limited': [False, False]},

          {'value': 1.0,
           'fixed': False,
           'parname': 'y1',
           'limited': [False, False]},

          {'value': 1.0,
           'fixed': False,
           'parname': 'y2',
           'limited': [False, False]},

          {'value': 1.0,
           'fixed': False,
           'parname': 'y3',
           'limited': [False, False]} ]

          
#=============================================================================#
import os, sys, shutil
import math as m
import numpy as np
import matplotlib as mpl
import pylab as pl
from mpfit import mpfit


#-----------------------------------------------------------------------------#
def main():

    # Generate a noisy polynomial
    #     [off,  x1,  x2,  x3,  y1,  y2,  y3]
    pIn = [2.0, 1.5, 0.1, 0.3, 1.0, 2.0, 0.05]
    pIn = [1.0, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0]
    shape = (200, 200)
    X, Y, Z, xyData = genpolydata(pIn, shape, 300, 10.2)

    # Define an function to evaluate the residual
    def errFn(p, fjac=None):
        status = 0
        # poly_surface' returns the 'rfunc' function and the X,Y data is
        # inserted via argument unpacking.
        return status, poly_surface(p)(*[Y, X]) - Z
    
    # Fit the data starting from an initial guess
    mp = mpfit(errFn, parinfo=inParms, quiet=False)
    print() 
    for i in range(len(inParms)):
        print("%s = %f +/- %f" % (inParms[i]['parname'],
                                  mp.params[i],
                                  mp.perror[i]))
    p1 = mp.params

    #-------------------------------------------------------------------------#
  
    # Plot the original, fit & residual
    fig = pl.figure(figsize=(18,4.3))
    
    ax1 = fig.add_subplot(1,3,1)
    cax1 = ax1.imshow(xyData, origin='lower',cmap=mpl.cm.jet)
    cbar1=fig.colorbar(cax1, pad=0.0)
    ax1.scatter(X, Y, c=Z, s=40, cmap=mpl.cm.jet)
    ax1.set_title("Sampled Data")
    ax1.set_xlim(0, shape[-1]-1)
    ax1.set_ylim(0, shape[-2]-1)
    ax1.set_aspect('equal')
    
    ax2 = fig.add_subplot(1,3,2)
    xyDataFit = poly_surface(p1, shape)
    cax2 = ax2.imshow(xyDataFit, origin='lower', cmap=mpl.cm.jet)
    cbar2=fig.colorbar(cax2, pad=0.0)
    ax2.set_title("Model Fit")
    
    ax3 = fig.add_subplot(1,3,3)
    xyDataRes = xyData - xyDataFit
    cax3 = ax3.imshow(xyDataRes, origin='lower', cmap=mpl.cm.jet)
    cbar2=fig.colorbar(cax3, pad=0.0)
    ax3.set_title("Residual")
    
    pl.show()

#-----------------------------------------------------------------------------#
def poly_surface(params, shape=None):

    p = params
    
    def rfunc(y, x):

        z = p[0] + (p[1]*x + p[2]*x**2.0 + p[3]*x**3.0 +
                    p[4]*y + p[5]*y**2.0 + p[6]*y**3.0)
        return z
    
    if shape is not None:
        return rfunc(*np.indices(shape))
    else:
        return rfunc


#-----------------------------------------------------------------------------#
def genpolydata(params, shape, nSamps=300, noiseFrac=0.2):
    
    # Generate a noisy gaussian image
    xyData = poly_surface(params, shape)
    xyData += (np.random.random(xyData.shape) - 0.5) * noiseFrac
    
    # Sample the data at discrete pixels
    X = np.random.random(nSamps) * xyData.shape[-1] -1
    X = np.array(np.round(X), dtype='int')
    Y = np.random.random(nSamps) * xyData.shape[-2] -1
    Y = np.array(np.round(Y), dtype='int')
    Z = xyData[Y, X]

    return X, Y, Z, xyData

#-----------------------------------------------------------------------------#
main()
