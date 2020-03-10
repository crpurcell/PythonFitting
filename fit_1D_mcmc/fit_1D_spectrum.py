#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     mk_line_mcmc.py                                                   #
#                                                                             #
# PURPOSE:  Simple code to run MCMC on a straight-line fit.                   #
#                                                                             #
# MODIFIED: 18-Jan-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

# Input data
datFile = 'spectrum.dat'

# Sampler parameters
nThreads = 4
nWalkers = 100
nSteps = 1000
nBurnSteps = 100

# Plot the walker positions?
doPlots = True

#=============================================================================#

import sys
import math as m
import numpy as np
import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import emcee


#-----------------------------------------------------------------------------#
def main():

    # Read in the ASCII Data
    xArr, yArr, dyArr = np.loadtxt(datFile, unpack=True)

    # We are fitting to the log of the data
    xArr = np.log10(xArr)
    yArr = np.log10(yArr)

    # Initialise the 'walkers' with random parameters around some guesses
    ndim = 2
    p0 = [np.random.rand(ndim) for i in range(nWalkers)]

    # Define an MCMC sampler object
    sampler = emcee.EnsembleSampler(nWalkers, ndim, lnprob,
                                    args=[xArr, yArr, dyArr],
                                    threads=nThreads)

    # Run it for a burn-in phase
    print('> Burning-in walkers ...', end="")
    sys.stdout.flush()
    pos, prob, state = sampler.run_mcmc(p0, nBurnSteps, storechain=True)
    #sampler.reset()
    print('done.')

    # Plot the burn-in chains
    if doPlots:
        print('> Plotting the walker chains during burn-in phase.')
        labels = ['m', 'c']
        fig1 = pl.figure(figsize=(7.5, 10))
        for j in range(ndim):
            ax = fig1.add_subplot(ndim,1,j+1)
            ax.plot(np.array([sampler.chain[:,i,j] for i in range(nBurnSteps)]),
                    'k', alpha = 0.3)
            ax.set_ylabel(labels[j], fontsize = 15)
        plt.xlabel('Steps', fontsize = 15)
        fig1.show()

    # Run sampler in earnest
    print('> Running MCMC samplers in earnest ...', end="")
    sys.stdout.flush()
    sampler.reset()
    sampler.run_mcmc(pos, nSteps, storechain=True)
    print('done.')

    # Plot the  chains
    if doPlots:
        print('> Plotting the walker chains.')
        fig2 = pl.figure(figsize=(7.5, 10))
        for j in range(ndim):
            ax = fig2.add_subplot(ndim,1,j+1)
            ax.plot(np.array([sampler.chain[:,i,j] for i in range(nSteps)]),
                    'k', alpha = 0.3)
            ax.set_ylabel(labels[j], fontsize = 15)
        plt.xlabel('Steps', fontsize = 15)
        fig2.show()

    # Normalise the returned likelyhoods
    maxL = np.max(sampler.flatlnprobability)
    minL = np.min(sampler.flatlnprobability)
    rangeL = maxL - minL
    normL = (sampler.flatlnprobability - minL)/rangeL

    # Save the flattened chain for each parameter
#    np.savetxt('chains.dat', zip(range(len(sampler.flatchain[:,0])),
#                                 sampler.flatchain[:,0],
#                                 sampler.flatchain[:,1],
#                                 normL))

    # Plot the walker points
    if doPlots:
        print('> Plotting the likelihood space.')
        fig3 = pl.figure(figsize=(7.5, 7.5))
        ax = fig3.add_subplot(1,1,1)
        cax = ax.scatter(sampler.flatchain[:,1], sampler.flatchain[:,0],
                         c=normL, edgecolor='None')
        ax.set_ylabel(labels[0], fontsize = 15)
        ax.set_xlabel(labels[1], fontsize = 15)
        fig3.colorbar(cax)
        fig3.show()

    # Print the values
    mSamps = sampler.flatchain[:, 0]
    g = lambda v: (v[1], v[2]-v[1], v[1]-v[0])
    mBest, errPlus, errMinus = g(np.percentile(mSamps, [15.72, 50, 84.27]))
    print('m = %.4g (+%3g, -%3g)' % (mBest, errPlus, errMinus))
    bSamps = sampler.flatchain[:, 1]
    bBest, errPlus, errMinus = g(np.percentile(bSamps, [15.72, 50, 84.27]))
    print('b = %.4g (+%3g, -%3g)' % (bBest, errPlus, errMinus))

    # Plot the fit using the mean of the likelihood distributions
    if doPlots:
        fig4 = pl.figure(figsize=(7.5, 7.5))
        ax = fig4.add_subplot(1,1,1)
        ax.errorbar(xArr,yArr, yerr=dyArr, fmt="None")
        ax.plot(xArr, model(mBest, bBest, xArr))
        ax.legend(['Best Fit','Data'])
        ax.set_title('MCMC Hammer Fit to a Simple Function')
        ax.set_xlabel('logFreq.')
        ax.set_ylabel('logS')
        fig4.show()
    input('>> Press <RETURN> to exit ...')


#-----------------------------------------------------------------------------#
def model(m, b, x):
    '''A straight line'''

    return m * x + b


#-----------------------------------------------------------------------------#
def lnprob(walker, x, y, dy):
    '''Log-likelihood for the model'''

    chi2 = np.sum( ( model(walker[0], walker[1], x) - y) **2 / dy**2 )

    return - chi2 / 2.0


#-----------------------------------------------------------------------------#
main()
