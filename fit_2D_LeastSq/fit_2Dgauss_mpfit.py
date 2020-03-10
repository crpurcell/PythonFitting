#!/usr/bin/env python

# Initial model parameters
inParms=[ {'value': 5.1,
           'fixed': False,
           'parname': 'amp',
           'limited': [False, False]},
          
          {'value': 70.0,
           'fixed': False,
           'parname': 'X0',
           'limited': [False, False]},
          
          {'value': 70.0,
           'fixed': False,
           'parname': 'Y0',
           'limited': [False, False]},
          
          {'value': 10.0,
           'fixed': False,
           'parname': 'min',
           'limited': [False, False]},
          
          {'value': 10.0,
           'fixed': False,
           'parname': 'maj',
           'limited': [False, False]},
          
          {'value':  0.0,
           'fixed': False,
           'parname': 'pa',
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

    # Generate a noisy gaussian
    #     [amp, X0,     Y0,   min,  maj,  pa             ]
    pIn = [1.0, 100.0, 100.0, 20.0, 30.0, 60.0]
    shape = (200, 200)
    X, Y, Z, xyData = gengaussdata(pIn, shape, 300, 0.2)

    # Define an function to evaluate the residual
    def errFn(p, fjac=None):
        status = 0
        # twodgaussian' returns the 'gauss' function and the X,Y data is
        # inserted via argument unpacking.
        return status, twodgaussian(p)(*[Y, X]) - Z
    
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
    xyDataFit = twodgaussian(p1, shape)
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
def twodgaussian(params, shape=None):

    amp, xo, yo, cx, cy, pa = params

    pa = m.radians(pa)
    
    def rfunc(y, x):
        st = m.sin(pa)**2
        ct = m.cos(pa)**2
        s2t = m.sin(2*pa)
        a = (ct/cx**2 + st/cy**2)/2
        b = s2t/4 *(1/cy**2-1/cx**2)
        c = (st/cx**2 + ct/cy**2)/2
        v = amp*np.exp(-1*(a*(x-xo)**2 + 2*b*(x-xo)*(y-yo) + c*(y-yo)**2))
        return v
    
    if shape is not None:
        return rfunc(*np.indices(shape))
    else:
        return rfunc


#-----------------------------------------------------------------------------#
def gengaussdata(params, shape, nSamps=300, noiseFrac=0.2):
    
    # Generate a noisy gaussian image
    xyData = twodgaussian(params, shape)
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
