#!/usr/bin/env python
#

from numpy import *
from scipy import optimize
from pylab import *



def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = sqrt(abs((arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = sqrt(abs((arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: ravel(gaussian(*p)(*indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    print(p, success)
    return p




# Create the gaussian data (with a bit of noise:

Xin, Yin = mgrid[0:401, 0:401]
data = gaussian(3, 200, 200, 40, 80)(Xin, Yin) + random(Xin.shape)

matshow(data, cmap=cm.jet)

params = fitgaussian(data)
fit = gaussian(*params)

contour(fit(*indices(data.shape)), cmap=cm.Greys)
ax = gca()
(height, x, y, width_x, width_y) = params

text(0.95, 0.05, """
x : %.1f
y : %.1f
width_x : %.1f
width_y : %.1f """ %(x, y, width_x, width_y),
        fontsize=16, color='w', horizontalalignment='right',
        verticalalignment='bottom', transform=ax.transAxes)


title('Fitting a 2-d Gaussian', fontsize=16, color='blue')

show()

