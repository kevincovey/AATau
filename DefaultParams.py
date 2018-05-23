#ensure that print and division functions work correctly under python 2.X
from __future__ import print_function, division 

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
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 4
plt.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


