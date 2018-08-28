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

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

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
    
def get_model(temp, logg):

    model_file = '/Volumes/CoveyData/StarSpotModelling/Med_Res_10K_grid/lte'+temp+'-'+logg+'-0.0.PHOENIX-ACES-AGSS-COND-2011-R-3500.fits'
    hdulist = fits.open(model_file)

    data = hdulist[1].data
        
    flux = data.field('FLUX')
    wavelength = data.field('LAMBDA')/10000.
    
    #wavelength = np.exp(hdulist[0].header[('CRVAL1')]+hdulist[0].header[('CDELT1')]*np.arange(0,212027))
    return wavelength, flux, temp, hdulist

def measure_UVlines(w, flux):

    #define the number of lines we're measuring
    n_lines = 3
    
    #make a 14 element array to keep EqWs measurements in
    EqWs = np.zeros(n_lines)

    #another array to keep line fluxes
    fluxes = np.zeros(n_lines)

    #save the boundaries of each line in 14 element arrays
    lines = np.zeros( (n_lines,2) )
    cont_red = np.zeros( (n_lines,2) )
    cont_blue = np.zeros( (n_lines,2) )

    lineIDs = ['G band']
    lineCenters = np.zeros(n_lines)

    #G Band
    lines[0,:] = [0.415, 0.425]
    cont_blue[0,:] = [0.455, 0.465]
    cont_red[0,:] = [0.455, 0.465] 
    lineCenters[0] = 0.42

    #Mg I 5172 Band (from Covey 2007)
    lines[1,:] = [0.51527, 0.51927]
    cont_blue[1,:] = [0.51, 0.515]
    cont_red[1,:] = [0.51, 0.515] 
    lineIDs.append('Mg I 5172')
    lineCenters[1] = 0.5172    

    #Modified version of Greg's R5150 index
    lines[2,:] = [0.505, 0.515]
    cont_blue[2,:] = [0.46, 0.47]
    cont_red[2,:] = [0.534, 0.544] 
    lineIDs.append('~R5150')
    lineCenters[2] = 0.51    
    
    for i in range(n_lines):
        this_eqw, this_lineflux, this_blah1, this_blah2 = measureEqW(w, flux, lines[i, :], cont_blue[i, :], cont_red[i, :], method = 'median')
        EqWs[i] = this_eqw
        fluxes[i] = this_lineflux

    return EqWs, fluxes, lineIDs, lineCenters


def measure_Optlines(w, flux):

    #define the number of lines we're measuring
    n_lines = 5
    
    #make a 14 element array to keep EqWs measurements in
    EqWs = np.zeros(n_lines)

    #another array to keep line fluxes
    fluxes = np.zeros(n_lines)

    #save the boundaries of each line in 14 element arrays
    lines = np.zeros( (n_lines,2) )
    cont_red = np.zeros( (n_lines,2) )
    cont_blue = np.zeros( (n_lines,2) )

    lineIDs = ['TiO 6250']
    lineCenters = np.zeros(n_lines)
    
    #TiO 6250
    lines[0,:] = [0.624, 0.627]
    cont_blue[0,:] = [0.643, 0.6465]
    cont_red[0,:] = [0.643, 0.6465] 
    lineIDs.append('TiO 6250')
    lineCenters[0] = 0.6255    

    #TiO 6800
    lines[1,:] = [0.675, 0.69]
    cont_blue[1,:] = [0.66,0.666]
    cont_red[1,:] = [0.699, 0.705] 
    lineIDs.append('TiO 6800')
    lineCenters[1] = 0.6825    

    #TiO 7140
    lines[2,:] = [0.713, 0.7155]
    cont_blue[2,:] = [0.7005, 0.7035]
    cont_red[2,:] = [0.7005, 0.7035] 
    lineIDs.append('TiO 7140')
    lineCenters[2] = 0.71425

    #TiO 7700
    lines[3,:] = [0.775, 0.78]
    cont_blue[3,:] = [0.812, 0.816]
    cont_red[3,:] = [0.812, 0.816]
    lineIDs.append('TiO 7700')
    lineCenters[3] = 0.7775

    #TiO 8465
    lines[4,:] = [0.8455, 0.8475]
    cont_blue[4,:] = [0.8345, 0.8385]
    cont_red[4,:] = [0.8345, 0.8385] 
    lineIDs.append('TiO 8465')
    lineCenters[4] = 0.8465
    
    for i in range(n_lines):
        this_eqw, this_lineflux, this_blah1, this_blah2 = measureEqW(w, flux, lines[i, :], cont_blue[i, :], cont_red[i, :], method = 'median')
        EqWs[i] = this_eqw
        fluxes[i] = this_lineflux

    return EqWs, fluxes, lineIDs, lineCenters


def measure_Jlines(w, flux):

    #define the number of lines we're measuring
    n_lines = 9
    
    #make a 14 element array to keep EqWs measurements in
    EqWs = np.zeros(n_lines)

    #another array to keep line fluxes
    fluxes = np.zeros(n_lines)

    #save the boundaries of each line in 14 element arrays
    lines = np.zeros( (n_lines,2) )
    cont_red = np.zeros( (n_lines,2) )
    cont_blue = np.zeros( (n_lines,2) )

    lineIDs = ['Ca 1.03']
    lineCenters = np.zeros(n_lines)

    #Ca 1.03 [1.0348,0.0015,1.030,0.005,1.0375,0.003]
    lines[0,:] = [1.03405, 1.03555]
    cont_blue[0,:] = [1.0275, 1.0325]
    cont_red[0,:] = [1.036, 1.039] 
    #lineIDs.append('Ca 1.03')
    lineCenters[0] = 1.0348
    
    #Si 1.06 [1.05888,0.0013,1.055,0.005,1.064,0.004]
    lines[1,:] = [1.05823, 1.05953]
    cont_blue[1,:] = [1.0525, 1.0575]
    cont_red[1,:] = [1.062,1.066] 
    lineIDs.append('Si 1.06')
    lineCenters[1] = 1.05888
    
    #Al 1.12
    #[1.1258,0.0015,1.1225,0.005,1.1295,0.005]
    lines[2,:] = [1.12505, 1.12655]
    cont_blue[2,:] = [1.12, 1.125]
    cont_red[2,:] = [1.12925,1.12975] 
    lineIDs.append('Al 1.12')
    lineCenters[2] = 1.1258
    
    #'Na1.14','Fe1.17','K1.18','Mg1.18',

    #Fe 1.17
    #[1.1645,0.011,1.155,0.005,1.1733,0.004]
    lines[3,:] = [1.159, 1.170]
    cont_blue[3,:] = [1.1525, 1.1575]
    cont_red[3,:] = [1.1713,1.1753] 
    lineIDs.append('Fe 1.17')
    lineCenters[3] = 1.1645


    #'Fe1.19',
    #[1.1887,0.0013,1.186,0.004,1.1925,0.004]
    lines[4,:] = [1.18805, 1.18935]
    cont_blue[4,:] = [1.184, 1.188]
    cont_red[4,:] = [1.1905,1.1945] 
    lineIDs.append('Fe 1.19')
    lineCenters[4] = 1.1887

    #'Fe1.20','K1.24','K1.25',
    #'Na1.27','Mn1.29','Al1.31',
    
    #[1.1395,0.0042,1.131,0.005,1.146,0.002], [1.1645,0.011,1.155,0.005,1.1733,0.004], [1.1776,0.0014,1.1727,0.0044,1.181625,0.00175], [1.1832,0.0014,1.181625, 0.00175,1.1865,0.003],
    #[1.1887,0.0013,1.186,0.004,1.1925,0.004], [1.19765,0.0012,1.1925,0.0045,1.20175,0.0025],


    #K1.24
    #[1.24368,0.0012,1.239,0.005,1.249,0.005],
    lines[5,:] = [1.24308, 1.24428]
    cont_blue[5,:] = [1.2365, 1.2415]
    cont_red[5,:] = [1.2465,1.2515] 
    lineIDs.append('K 1.24')
    lineCenters[5] = 1.24368
    
    #K1.25
    #[1.2526,0.0012,1.2495,0.004,1.256,0.004]
    lines[6,:] = [1.252, 1.2532]
    cont_blue[6,:] = [1.2475, 1.2515]
    cont_red[6,:] = [1.254,1.258] 
    lineIDs.append('K 1.25')
    lineCenters[6] = 1.2526
    
    #[1.26821,0.0025,1.263,0.001,1.27175,0.0025],

    #Mn 1.29
    #[1.29045,0.0013,1.28675,0.002,1.2961,0.003],
    lines[7,:] = [1.2898, 1.2911]
    cont_blue[7,:] = [1.28525, 1.28825]
    cont_red[7,:] = [1.2946,1.2976] 
    lineIDs.append('Mn 1.29')
    lineCenters[7] = 1.29045
    
    #Al 1.31
    #[1.3141,0.0042,1.307,0.004,1.321,0.004],
    lines[8,:] = [1.312, 1.3162]
    cont_blue[8,:] = [1.305, 1.309]
    cont_red[8,:] = [1.319,1.323] 
    lineIDs.append('Al 1.31')
    lineCenters[8] = 1.3141
    
    for i in range(n_lines):
        this_eqw, this_lineflux, this_blah1, this_blah2 = measureEqW(w, flux, lines[i, :], cont_blue[i, :], cont_red[i, :], method = 'median')
        EqWs[i] = this_eqw
        fluxes[i] = this_lineflux

    return EqWs, fluxes, lineIDs, lineCenters
  
#measure features from Table 3 of Covey 2010 for spectral typing purposes. 
def measure_Covey2010lines(w, flux):

    #define the number of lines we're measuring
    n_lines = 14
    
    #make a 14 element array to keep EqWs measurements in
    EqWs = np.zeros(n_lines)

    #another array to keep line fluxes
    fluxes = np.zeros(n_lines)

    #save the boundaries of each line in 14 element arrays
    lines = np.zeros( (n_lines,2) )
    cont_red = np.zeros( (n_lines,2) )
    cont_blue = np.zeros( (n_lines,2) )

    lineIDs = ['Mg 1.50']
    lineCenters = np.zeros(n_lines)
    
    #mg1.50
    lines[0,:] = [1.502, 1.506]
    cont_blue[0,:] = [1.49575, 1.50025]
    cont_red[0,:] = [1.50725, 1.51175] 
    lineCenters[0] = 1.504
    
    #k1.52
    lines[1,:] = [1.5152, 1.5192]
    cont_blue[1,:] = [1.5085, 1.5125]
    cont_red[1,:] = [1.521, 1.525] 
    lineIDs.append('KI 1.51') 
    lineCenters[1] = 1.5172
    
    #mg1.58
    lines[2, :] = [ 1.574 , 1.578 ]
    cont_blue[2, :] = [ 1.568 , 1.572 ]
    cont_red[2, :] = [ 1.578 , 1.582 ]
    lineIDs.append('Mg I 1.58')
    lineCenters[2] = 1.576

    #si1.59
    lines[3, :] = [ 1.5875 , 1.5925 ]
    cont_blue[3, :] = [ 1.5845 , 1.5875 ]
    cont_red[3, :] = [ 1.5925 , 1.5955 ]
    lineIDs.append('Si I 1.59')
    lineCenters[3] = 1.59
    
    #CO1.62
    lines[4, :] = [ 1.6175 , 1.622 ]
    cont_blue[4, :] = [ 1.6135 , 1.6165 ]
    cont_red[4, :] = [ 1.6265 , 1.6295 ]
    lineIDs.append('CO 1.62')
    lineCenters[4] = 1.61975
    
    #CO1.66
    lines[5, :] = [ 1.661 , 1.664 ]
    cont_blue[5, :] = [ 1.655 , 1.658 ]
    cont_red[5, :] = [ 1.6685 , 1.6715 ]
    lineIDs.append('CO 1.66')
    lineCenters[5] = 1.66
    
    #Al1.67
    lines[6, :] = [ 1.671 , 1.677 ]
    cont_blue[6, :] = [ 1.6575 , 1.6605 ]
    cont_red[6, :] = [ 1.677 , 1.679 ]
    lineIDs.append('Al I 1.67')
    lineCenters[6] = 1.674

    #dip1.70
    lines[7, :] = [ 1.706 , 1.709 ]
    cont_blue[7, :] = [ 1.7025 , 1.7055 ]
    cont_red[7, :] = [ 1.713 , 1.716 ]
    lineIDs.append('dip 1.70')
    lineCenters[7] = 1.7075

    #Mg1.71
    lines[8, :] = [ 1.71 , 1.713 ]
    cont_blue[8, :] = [1.7025 , 1.7055 ]
    cont_red[8, :] = [ 1.713 , 1.716 ]
    lineIDs.append('Mg 1.71')
    lineCenters[8] = 1.7115

    #Ca1.98
    lines[9, :] = [ 1.9755 , 1.9885 ]
    cont_blue[9, :] = [ 1.9651 , 1.9701 ]
    cont_red[9, :] = [ 1.99525 , 2.00025 ]
    lineIDs.append('Ca 1.98')
    lineCenters[9] = 1.982

    #Na2.21
    lines[10, :] = [ 2.204 , 2.211 ]
    cont_blue[10, :] = [ 2.193 , 2.197 ]
    cont_red[10, :] = [ 2.214 , 2.22 ]
    lineIDs.append('Na 2.21')
    lineCenters[10] = 2.2075

    #Ca2.26
    lines[11, :] = [ 2.2605 , 2.2675 ]
    cont_blue[11, :] = [ 2.25575 , 2.26025 ]
    cont_red[11, :] = [ 2.26775 , 2.27225]
    lineIDs.append('Ca 2.26')
    lineCenters[11] = 2.264

    #CO2.3
    lines[12, :] = [ 2.2925 , 2.305 ]    #-- this one has been modified!!!
#    lines[12, :] = [ 2.2925 , 2.315 ]    #-- this one has been modified!!!
    cont_blue[12, :] = [ 2.2845 , 2.2915 ]
    cont_red[12, :] = [ 2.3065 , 2.3105]
#    cont_red[12, :] = [ 2.3165 , 2.3205]
    lineIDs.append('CO 2.3')
    lineCenters[12] = 2.30375

    #CO2.34
    lines[13, :] = [ 2.344 , 2.347 ]
    cont_blue[13, :] = [ 2.341 , 2.344 ]
    cont_red[13, :] = [ 2.3475, 2.3505 ]
    lineIDs.append('CO 2.34')
    lineCenters[13] = 2.3455

    for i in range(n_lines):
        this_eqw, this_lineflux, this_blah1, this_blah2 = measureEqW(w, flux, lines[i, :], cont_blue[i, :], cont_red[i, :], method = 'median')
        EqWs[i] = this_eqw
        fluxes[i] = this_lineflux

    return EqWs, fluxes, lineIDs, lineCenters
    
