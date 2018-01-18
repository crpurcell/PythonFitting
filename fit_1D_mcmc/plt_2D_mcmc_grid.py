#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     plt_2D_mcmc.py                                                    #
#                                                                             #
# PURPOSE:  Plot the 1 and 2D confidence distributions for two parameters     #
#           output by the MCMC python script                                  #
#                                                                             #
# MODIFIED: 05-May-2014 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

# Input MCMC ing data (2D array of points)
inFile = 'chains.dat'

# Labels
xLabel = r'Slope'
yLabel = r'Intercept'

# Plot type [scatter, hist2D, hexbin]
plotType = 'scatter'
plotType = 'hist2D'
#plotType = 'hexbin'

# Number of averaging bins
nXbins = 50
nYbins = 50


#=============================================================================#
import sys
import numpy as np
import matplotlib as mpl
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from scipy import ndimage as ndi
from scipy.interpolate import interp1d
from scipy.ndimage.filters import median_filter


#-----------------------------------------------------------------------------#
def main():

    # Load the data
    nStep, xData, yData, zData = np.loadtxt(inFile, unpack=True)

    # Draw the confidence plot
    make_confidence_plot(xData, yData, zData, nXbins, nYbins)
    
    
#-----------------------------------------------------------------------------#
def make_confidence_plot(xData, yData, zData, nXbins, nYbins):

    # Data limits
    xMinData = min(xData)
    xMaxData = max(xData)
    yMinData = min(yData)
    yMaxData = max(yData)

    # Bin the XY-data
    xBins = np.linspace(xMinData, xMaxData, int(nXbins+1))
    xBinWidth = (xMaxData - xMinData)/nXbins
    yBins = np.linspace(yMinData, yMaxData, int(nYbins+1))
    yBinWidth = (yMaxData - yMinData)/nYbins
    nXY, yBins, xBins = np.histogram2d(yData, xData, (yBins, xBins))
    nX, xBins =  np.histogram(xData, xBins)  # This preempts 'ax.hist' so
    nY, yBins =  np.histogram(yData, yBins)  # we can determine y-limits.

    # Print the mean & error from the likelihood distribution
    mX = get_maxLparms(nX, xBins)
    print " Slope = %.2f, (dNeg = %.2f, dPos = %.2f)" % \
          (    mX['mean'],
               (mX['mean']-
                mX['dMmin']),
               (mX['dMmax']-
                mX['mean']))
    mY = get_maxLparms(nY, yBins)
    print " Intercept = %.2f, (dNeg = %.2f, dPos = %.2f)" % \
          (    mY['mean'],
               (mY['mean']-
                mY['dMmin']),
               (mY['dMmax']-
                mY['mean']))
    
    #-------------------------------------------------------------------------#
    
    # Setup the figure page
    fig=pl.figure(figsize=(9.6, 9.0))

    # Alter the default linewidths etc
    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['axes.linewidth'] = 1.0
    mpl.rcParams['xtick.major.size'] = 6.0
    mpl.rcParams['xtick.minor.size'] = 4.0
    mpl.rcParams['ytick.major.size'] = 6.0
    mpl.rcParams['ytick.minor.size'] = 4.0
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 14.0

    # Plot the density
    ax1 = fig.add_subplot(111)

    # Plot as a density map
    if plotType=='hist2D':
        im = ax1.imshow(nXY, interpolation='nearest', origin='lower',
                          extent=[xMinData,xMaxData,yMinData,yMaxData],
                          aspect='auto', cmap='jet',
                          norm=LogNorm())
    # ... or as a hexagon density map
    elif plotType=='hexbin':
        im = ax1.hexbin(xData, yData, gridsize=(nXbins, nYbins), bins='log',
                          cmap='jet')

    # ... or a scatter plot
    else:
        im =  ax1.scatter(xData, yData, c=zData, s=20, edgecolor='None')

    # Plot the contours
    maxDensity = np.max(nXY)
    levels = get_sigma_levels(nXY)
    cax2 = ax1.contour(nXY, levels=levels, colors='k', lw=2.0,
                       extent=[xMinData,xMaxData,yMinData,yMaxData])

    # Plot the 1-D histograms
    divider = make_axes_locatable(ax1)
    ax2 = divider.append_axes("top", 2.0, pad=0.2, sharex=ax1)
    ax3 = divider.append_axes("right", 2.0, pad=0.2, sharey=ax1)
    
    # Set the ajoining labels to invisible
    pl.setp(ax2.get_xticklabels() + ax3.get_yticklabels(), visible=False)
    
    # Plot the histograms
    yMaxPlot = float(max(nX))*1.2
    ax2.hist(xData, bins=xBins,histtype='stepfilled', color='silver', lw=1.0)
    ax3.hist(yData, bins=yBins, histtype='stepfilled', color='silver',
             orientation='horizontal', lw=1.0)
    
    # Set the plot limits
    #ax1.set_xlim(xMinData, xMaxData)
    #ax1.set_ylim(yMinData, yMaxData)
    ax2.set_ylim(0.001, max(nX)*1.15)
    ax3.set_xlim(0.001, max(nY)*1.15)
    
    # Limit the number of ticks
    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax3.xaxis.set_major_locator(MaxNLocator(4))

    # Formatting
    ax1.set_xlabel(xLabel)
    ax1.set_ylabel(yLabel)
    ax2.set_ylabel('Sample density')
    ax3.set_xlabel('Sample density')
    format_ticks(ax1, 10, 1.2)
    format_ticks(ax2, 10, 1.2)
    format_ticks(ax3, 10, 1.2)

    # Plot a colourbar
    cax = divider.append_axes("right", 0.2, pad=0.2,)
    cbar = fig.colorbar(im, cax=cax)
    if plotType=='hist2D' or plotType=='hexbin':
        cbar.set_label('2D Sample Density')        
    else:
        cbar.set_label('Normalised Likelihood')
    
    pl.show()

    
#-----------------------------------------------------------------------------#
def get_maxLparms(n, b, doMedFilt=True, doSmFilt=True):
    '''Measure the parameters from a 1D histogram of likelihood. Returns a
    dictionary containing the results.'''
    
    # Bin the population
    bWidth = np.diff(b)[0]
    bMid = (b + bWidth/2.0)[:-1]
    norm = n.sum()
    n = n.astype('float')/norm
    
    # Median filter PDF and smooth
    nSm = n
    if doMedFilt:
        nSm = ndi.filters.median_filter(nSm, size=3)
    if doSmFilt:
        nSm = ndi.gaussian_filter(nSm, 1)
        
    # Sort and integrate to 1-sigma mark
    bMsk = np.zeros_like(b).astype('bool')
    iSrt = np.argsort(nSm)
    runSum = 0.0 
    for i in iSrt[::-1]:
        runSum += nSm[i]
        if runSum/np.sum(nSm)<0.682689492:
            bMsk[i] = True

    # Calculate the max L and +/- 1sigma limits
    # Calculate the mean and stdev
    pDict = {}
    iLt = np.argwhere(bMsk).flatten()[0]
    iRt = np.argwhere(bMsk).flatten()[-1]
    iMax = np.argmax(nSm)
    pDict['maxL'] = bMid[iMax]
    pDict['dLmin'] = bMid[iLt]
    pDict['dLmax'] = bMid[iRt]
    
    # Calculate the cumulative hisogram
    # Calculate the mean and stdev
    nCumSm = np.cumsum(nSm)
    f = interp1d(nCumSm, bMid)
    pDict['mean'] = f(0.5)
    try:
        pDict['dMmin'] = f(0.1572)
    except:
        pDict['dMmin'] = 0.0
    try:
        pDict['dMmax'] = f(0.8427)
    except:
        pDict['dMmax'] = 0.0    

    return pDict


#-----------------------------------------------------------------------------#
def get_sigma_levels(img, cLevels=[0.682689492, 0.954499736, 0.997300204]):
    '''Determine the contour levels at which 1, 2 and 3 sigma of the data are
    enclosed.'''
    
    ind = np.unravel_index(np.argsort(img, axis=None)[::-1], img.shape)
    cumsum = np.cumsum(img[ind])/np.sum(img[ind])
    aLevels = []
    for cLevel in cLevels:
        aLevels.append(img[ind][np.where(cumsum<cLevel)[0][-1]])
        
    return aLevels

    
#-----------------------------------------------------------------------------#
def format_ticks(ax, pad=10, w=1.0):
    '''Format the ticks on a matplotlib plot'''
    
    ax.tick_params(pad=pad)
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markeredgewidth(w)


#-----------------------------------------------------------------------------#
main()
