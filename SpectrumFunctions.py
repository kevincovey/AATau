#import numpy, scipy, math and astropy libraries
import numpy as np
import scipy as sp
import math
import astropy

#import various astropy functionalities
#is there a benefit/difference between these two import mechanisms?
import astropy.units as u
from astropy.io import fits
from astropy.table import Table

#import various other functionalities (again, confused by the difference in these import statements)
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from sys import platform

#import matplotlib so that we can make plots; import common subroutines and give them compact names
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.colorbar as cb

#set plotting defaults to values that make plots look publication ready
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 8, 5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 4
plt.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


def measureEqW(w, flux, line, blue_cont, red_cont, method = 'median' ):
    '''cut out and normalize flux around a line,
    stolen from http://www.astropy.org/astropy-tutorials/UVES.html

    Parameters
    ----------
    w : 1 dim np.ndarray
    array of wavelengths
    flux : np.ndarray flux values
    line : wavelength limits of the line to be integrated [blue_limit, red_limit]
    blue_cont : wavelength limits of the blue continuum region [blue_limit, red_limit]
    red_cont : wavelength limits of the red continuum region [blue_limit, red_limit]
    '''

    #index is true in the region where we measure the continuum
    in_cont = ((w > blue_cont[0]) & (w < blue_cont[1])) | ((w > red_cont[0]) & (w < red_cont[1]))

    #index of the region we want to measure the EqW of
    in_line = (w > line[0]) & (w < line[1])

    # make a flux array of shape
    # (number of spectra = 1, number of points in indrange)
    line_fluxes = flux[in_line]
    faux_continuum_fluxes = np.ones((in_line.sum()))
    
    #measure continuum using method of choice
    #first, use median in continuum region
    if method == 'median':
        continuum_median = np.median(flux[in_cont])
        faux_continuum_fluxes = faux_continuum_fluxes * continuum_median
    else:
        # fit polynomial of second order to the continuum region
        linecoeff = np.polyfit(w[in_cont], flux[in_cont],2)
        faux_continuum_fluxes = np.polyval(linecoeff, w[in_line])

    #normalize feature by continuum 
    normalized_line_fluxes = line_fluxes / faux_continuum_fluxes

    #integrate the normalized line fluxes after subtracting off 1
    EqW = np.trapz(normalized_line_fluxes - 1 , w[in_line])

    #integrate the line just after subtracting the continuum (i.e., preserve the flux units)
    FluxWidth = np.trapz(line_fluxes - faux_continuum_fluxes, w[in_line])
    
    return EqW, FluxWidth, w[in_line], normalized_line_fluxes

def readSpectrum(filename, units):
    '''read in a spectrum, save as an astropy table

    Parameters
    __________
    filename : name of the ascii file containing the spectrum (in wavelength and flux columns)
    units : type of units to assign to wavelengths  [edit later to include flux units?]
    '''

    #read in the spectrum
    spec = Table.read(filename, format = 'ascii', names = ['wav', 'flux'])

    #fill the arrays
    wav = np.array(spec['wav']) #*units
    flux=np.array(spec['flux'])

    return wav, flux
    
