#!/usr/bin/env python
from __future__ import print_function
#=============================================================================#
#                                                                             #
# NAME:     fit_1D_line_multinest.py                                          #
#                                                                             #
# PURPOSE:  Example of using PyMultiNest to fit a polynomial to some data     #
#                                                                             #
# MODIFIED: 30-Jan-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#
# Input dataset 
specDat = "polySpec.dat"

# Output directory for chains
outDir = specDat + "_out"

# Prior type and limits of parameters in 3rd order polynomial model
# Type can be "uniform", "normal", "log" or "fixed" (=set to boundsLst[n][1])
priorLst = [["uniform",   0.0,   2.0],    #   0 < p[0] < 2
            ["uniform",  -1.0,   1.0],    #  -1 < p[1] < 1 
            ["uniform",  -1.0,   1.0],    #  -1 < p[2] < 1 
            ["uniform",  -1.0,   1.0]]    #  -1 < p[4] < 1

# Number of points
nPoints = 1000

# Control verbosity
verbose = True
debug = False
showPlots = True

#=============================================================================#

import os
import sys
import shutil
import json
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import ndtri

import pymultinest as pmn
from Imports import corner

# Check if mpi4py is available
try:
    from mpi4py import MPI
    mpiSwitch = True
except:
    mpiSwitch = False
    

#-----------------------------------------------------------------------------#
def main():

    # Get the processing environment
    if mpiSwitch:
        mpiComm = MPI.COMM_WORLD
        mpiSize = mpiComm.Get_size()
        mpiRank = mpiComm.Get_rank()
    else:
        mpiRank = 0
        
    # Let's time the sampler
    if mpiRank==0:
        startTime = time.time()
    
    # Create the output directory    
    if mpiRank==0:
        if os.path.exists(outDir):
            shutil.rmtree(outDir, True)
        os.mkdir(outDir)
    if mpiSwitch:
        mpiComm.Barrier()
    
    # Read in the spectrum
    if mpiRank==0:
        specArr = np.loadtxt(specDat, dtype="float64", unpack=True)
    else:
        specArr = None
    if mpiSwitch:
        specArr = mpiComm.bcast(specArr, root=0)
    xArr = specArr[0] / 1e9   # GHz -> Hz for this dataset
    yArr = specArr[1]
    dyArr = specArr[4]

    # Set the prior function given the bounds of each parameter
    prior = prior_call(priorLst)
    nDim = len(priorLst)
        
    # Set the likelihood function
    lnlike = lnlike_call(xArr, yArr, dyArr)

    # Run nested sampling
    argsDict = init_mnest()
    argsDict["n_params"]             = nDim
    argsDict["n_dims"]               = nDim
    argsDict["outputfiles_basename"] = outDir + "/"
    argsDict["n_live_points"]        = nPoints
    argsDict["verbose"]              = verbose
    argsDict["LogLikelihood"]        = lnlike
    argsDict["Prior"]                = prior
    pmn.run(**argsDict)

    # Do the post-processing on one processor
    if mpiSwitch:
        mpiComm.Barrier()
    if mpiRank==0:
        
        # Query the analyser object for results
        aObj = pmn.Analyzer(n_params = nDim, outputfiles_basename=outDir + "/")
        statDict =  aObj.get_stats()
        fitDict =  aObj.get_best_fit()        
        endTime = time.time()

        # DEBUG
        if debug:
            print("\n", "-"*80)
            print("GET_STATS() OUTPUT")
            for k, v in statDict.iteritems():
                print("\n", k,"\n", v)

            print("\n", "-"*80)
            print("GET_BEST_FIT() OUTPUT"  )
            for k, v in fitDict.iteritems():
                print("\n", k,"\n", v)

        # Get the best fitting values and uncertainties
        p = fitDict["parameters"]
        lnLike = fitDict["log_likelihood"]
        lnEvidence = statDict["nested sampling global log-evidence"]
        dLnEvidence = statDict["nested sampling global log-evidence error"]
        med = [None] *nDim
        dp = [[None, None]]*nDim
        for i in range(nDim):   
            dp[i] = statDict["marginals"][i]['1sigma']
            med[i] = statDict["marginals"][i]['median']

        # Calculate goodness-of-fit parameters
        nSamp = len(xArr)
        dof = nSamp - nDim -1
        chiSq = calc_chisq(p, xArr, yArr, dyArr)
        chiSqRed = chiSq/dof
        AIC = 2.0*nDim - 2.0 * lnLike
        AICc = 2.0*nDim*(nDim+1)/(nSamp-nDim-1) - 2.0 * lnLike
        BIC = nDim * np.log(nSamp) - 2.0 * lnLike
        
        # Summary of run
        print("")
        print("-"*80)
        print("SUMMARY OF SAMPLING RUN:")
        print("NUM-PROCESSORS: %d" % mpiSize)
        print("RUN-TIME: %.2f" % (endTime-startTime))
        print("DOF:", dof)
        print("CHISQ:", chiSq)
        print("CHISQ RED:", chiSqRed)
        print("AIC:", AIC)
        print("AICc", AICc)
        print("BIC", BIC)
        print("ln(EVIDENCE)", lnEvidence)
        print("dLn(EVIDENCE)", dLnEvidence)
        print("")
        print('-'*80)
        print("BEST FIT PARAMETERS & MARGINALS:")
        for i in range(len(p)):
            print("p%d = %.4f [%.4f, %.4f]" % \
                  (i, p[i], dp[i][0], dp[i][1]))
    
        # Plot the data and best fit
        dataFig = plot_model(p, xArr, yArr, dyArr)
        dataFig.savefig(outDir + "/fig_best_fit.pdf")
    
        # Plot the triangle plot
        chains =  aObj.get_equal_weighted_posterior()
        cornerFig = corner.corner(xs = chains[:, :nDim],
                                  labels  = ["p" + str(i) for i in range(nDim)],
                                  range   = [0.99999]*nDim,
                                  truths  = p,
                                  bins    = 30)
        cornerFig.savefig(outDir + "/fig_corner.pdf")

        # Show the figures
        if showPlots:
            dataFig.show()
            cornerFig.show()
            print("Press <return> to continue ...", end="")
            raw_input()
    
        # Clean up
        plt.close(dataFig)
        plt.close(cornerFig)

    # Clean up MPI environment
    if mpiSwitch:
        MPI.Finalize()


#-----------------------------------------------------------------------------#
def model(p, x):
    """ Evaluate the model given an X array """
    
    return p[0] + p[1]*x + p[2]*x**2. + p[3]*x**3.


#-----------------------------------------------------------------------------#
def lnlike_call(xArr, yArr, dyArr):
    """ Returns a function to evaluate the log-likelihood """

    def lnlike(p, nDim, nParams):
        chisq = calc_chisq(p, xArr, yArr, dyArr)
        prefactor = np.sum( np.log(2.0*np.pi*dyArr**2.) )
        return -0.5*(prefactor + chisq)
                
    return lnlike


#-----------------------------------------------------------------------------#
def calc_chisq(p, xArr, yArr, dyArr):
    """ Calculate chi-squared for a model given the data """
    return np.sum( (yArr-model(p, xArr))**2./dyArr**2. )
    

#-----------------------------------------------------------------------------#
def prior_call(priorLst):
    """Returns a function to transform (0-1) range to the distribution of 
    values for each parameter. Note that a numpy vectorised version of this
    function fails because of type-errors."""

    def rfunc(p, nDim, nParams):
	for i in range(nDim):
            if priorLst[i][0] == "log":
		bMin = np.log(np.abs(priorLst[i][1]))
		bMax = np.log(np.abs(priorLst[i][2]))	
		p[i] *= bMax - bMin
		p[i] += bMin
		p[i] = np.exp(p[i])
            elif priorLst[i][0] == "normal":
                bMin, bMax = priorLst[i][1:]
                sigma = (bMax - bMin)/2.0
                mu = bMin + sigma
                p[i] = mu + sigma * ndtri(p[i])
            elif priorLst[i][0] == "fixed":
		p[i] = priorLst[i][1]
            else: # uniform (linear)
                bMin, bMax = priorLst[i][1:]
                p[i] = bMin + p[i] * (bMax - bMin)
        return p
    
    return rfunc


#-----------------------------------------------------------------------------#
def plot_model(p, x, y, dy, scaleX=1.0):

    # Make the model curve
    nSamples = 100
    dXSamp = (np.max(x) - np.min(x)) / nSamples
    iLst = np.arange(nSamples, dtype='float32')
    xSamp = np.min(x) + dXSamp * iLst
    ySamp = model(p, xSamp)

    # Plot the channels and fit
    fig = plt.figure()
    fig.set_size_inches([8,4])
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.plot(xSamp*scaleX, ySamp, color='b',marker='None',mfc='w',
            mec='g', ms=10, label='none', lw=1.0)
    ax.errorbar(x=x*scaleX , y=y, yerr=dy, mfc='none', ms=4, fmt='D',
                ecolor='red', elinewidth=1.0, capsize=2)

    return fig


#-----------------------------------------------------------------------------#
def init_mnest():
    """Initialise MultiNest arguments"""
    
    argsDict = {'LogLikelihood':              '',
                'Prior':                      '',
                'n_dims':                     0,
                'n_params':                   0,
                'n_clustering_params':        0,
                'wrapped_params':             None,
                'importance_nested_sampling': False,
                'multimodal':                 False,
                'const_efficiency_mode':      False,
                'n_live_points':              100,
                'evidence_tolerance':         0.5,
                'sampling_efficiency':        'model',
                'n_iter_before_update':       500,
                'null_log_evidence':          -1.e90,
                'max_modes':                  100,
                'mode_tolerance':             -1.e90,
                'outputfiles_basename':       '',
                'seed':                       -1,
                'verbose':                    True,
                'resume':                     True,
                'context':                    0,
                'write_output':               True,
                'log_zero':                   -1.e100,
                'max_iter':                   0,
                'init_MPI':                   False,
                'dump_callback':              None}
    return argsDict

    
#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
