#import numpy, scipy, math and astropy libraries
import numpy as np
import scipy as sp
import math
import astropy

# sum up over a region, with fractions of a pixel
# wavea = blue edge; waveb = red edge
# c1 and c2 are the edges of the region of interest
# mean returns the average of the pixels rather than the sum
# check makes plots
def fracsum(wavea, waveb, flux, c1, c2, mean=False, check=False):
  
    pixwidth = waveb-wavea  
    fracused = np.zeros_like(flux)
  
    # make sure wide enough
    if (wavea[0]>=c1) or (waveb[-1]<=c2):
      return None
    
    # whole pixels
    wholeind, = np.where( (wavea>=c1) & (waveb<=c2) )
    if len(wholeind) == 0:
      print("WARNING fracsum: no whole pixels")
    fracused[wholeind] = 1.
    
    # partial pixels
    lowind, = np.where( (wavea<c1) & (waveb>c1) )
    uppind, = np.where( (wavea<c2) & (waveb>c2) )
    assert(len(lowind) <= 1)
    assert(len(uppind) <= 1)

    if (len(lowind) == 1) and (not (lowind in wholeind)):
      fracused[lowind] = (waveb[lowind]-c1)/pixwidth[lowind]
    if (len(uppind) == 1) and (not np.any(uppind in wholeind)):
      fracused[uppind] = (c2-wavea[uppind])/pixwidth[uppind]

    if check:
      plt.scatter(wavea, flux)
      plt.scatter(waveb, flux)
      plt.plot([c1,c1],[0,10])
      plt.plot([c2,c2],[0,10])
      print(np.arange(0,len(flux)))
      print(fracused)
      print(flux)
    
    integral = np.sum(flux*fracused*pixwidth)
    if mean:
      print(flux, integral, np.sum(fracused*pixwidth), fracused, pixwidth)
  
      return integral/np.sum(fracused*pixwidth)
    else:
      return integral

# assume pixel edges are half-way between the central locations
def simple_fracsum(wave, flux, c1, c2, mean=False):
  
    diff = np.array(wave.flat[1:] - wave.flat[:-1])
    pixwidth = np.concatenate(([diff[0]], (diff[1:]+diff[:-1])/2., [diff[-1]]))

    wavea = wave-pixwidth/2.
    waveb = wave+pixwidth/2.
   
    return fracsum(wavea, waveb, flux, c1, c2, mean=mean)

# EW of arbitrary line     
def measure_eqw(wave, flux, line_start, line_end, blue_cont_start, blue_cont_end, 
               red_cont_start, red_cont_end, trapsum=False):
    
    # feature region
    feature = [line_start,line_end]
    
    m1 = simple_fracsum(wave, flux, blue_cont_start, blue_cont_end, mean=True)
    m2 = simple_fracsum(wave, flux, red_cont_start, red_cont_end, mean=True)
    pseudo = (m1+m2)/2.

    ew = simple_fracsum(wave, 1.-flux/pseudo, feature[0], feature[1])
    return ew
