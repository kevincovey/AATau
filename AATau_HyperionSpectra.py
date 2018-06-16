###### INITIAL STUFF ######

#import packages that we'll use often
import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline  <-- deprecate because scripting

#import the hyperion package
from hyperion.model import ModelOutput
from hyperion.util.constants import pc

#import our special spectral routines
#import SpectrumFunctions
import astropy.units as u

from SpectrumFunctions import get_model
from SpectrumFunctions import measure_Covey2010lines
from SpectrumFunctions import readSpectrum

#set up plotting defaults
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf','png')
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10,6
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 10


#### Read in AA Tau Model #######
#this model was computed by Hyperion using the AATau_example.py script,
#which depends on the AATau_example.rtin and kt04000g+3.5z-2.0.ascii files
AATau_mo = ModelOutput('Hyperion/AATau_example.rtout')

#### plot the AA Tau SED w/ mid-IR fluxes for comparison ####

#start by pulling in the SED info
AATau_sed = AATau_mo.get_sed(aperture=-1, distance = 137. * pc)  #<--- using distance estimate corresponding to Gaia DR2 parallax

#now define the number of inclinations we want to plot, and the color-map we want to use for them
inclinations_to_use = [0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18] 
number_of_inclinations_to_use = len(inclinations_to_use)
cmap = plt.get_cmap('RdBu')
colors = [cmap(i) for i in np.linspace(0, 1, number_of_inclinations_to_use)]

#now actually make plot of the SEDs
fig = plt.figure(figsize = (5,4))
ax = fig.add_subplot(1,1,1)
#for i in range(AATau_sed.val.shape[0]):
for i in range(number_of_inclinations_to_use):
    #print(i, inclinations_to_use[i])
    ax.loglog(AATau_sed.wav, AATau_sed.val[inclinations_to_use[i], :], color = colors[i], label = '$i$ = {:02.0f}'.format(inclinations_to_use[i]*5))

#put in a legend
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, bbox_to_anchor=(1.35, 1.25))

#now add some WISE data
lwise = np.array( [3.35, 4.6, 11.56, 22.09] )  # <-- WISE filtercenters in microns
lwise_widths = np.array( [6625, 10423, 55069, 41013] )  # <-- widths of WISE filters in ?Angstroms?
AATau_WISE = np.array( [8.59e-15, 4.77e-15, 9.03e-16, 6.41e-16] )  # <-- fluxes in erg/s/cm2/A
ax.loglog(lwise, AATau_WISE*lwise_widths, marker = 'o', color = 'black')

lneowise = np.array( [3.35, 4.6] )  # <-- WISE filter centers for just non-cyro bands
lneowise_widths = np.array( [6625, 10423] ) # <-- widths of filters for just non-cyro bands
AATau_NEOWISE = np.array( [1.56e-14, 9.18e-15] ) # <-- fluxes in erg/s/cm2/A
ax.loglog(lneowise, AATau_NEOWISE*lneowise_widths, marker = 'o', color = 'green')

#plot limits, etc.
ax.set_xlim(0.03, 2000.)
ax.set_ylim(2.e-15, 1e-8)
ax.set_title(r'AA Tau Model w/ mid-IR')
ax.set_xlabel(r'$\lambda$ [$\mu$m]')
ax.set_ylabel(r'$\lambda F_\lambda$ [ergs/cm$^2$]')
fig.savefig('Figures/AATau_midIR_sed.png', bbox_inches='tight')




#### plot the AA Tau polarization data ####

#start by pulling in the polarization info
AATau_sed_pol = AATau_mo.get_sed(stokes = 'linpol', aperture=-1)

#now actually make plot of the polarization data - start with the model
fig = plt.figure(figsize = (5,4))
ax = fig.add_subplot(1,1,1)
for i in range(number_of_inclinations_to_use):
    ax.loglog(AATau_sed_pol.wav, AATau_sed_pol.val[inclinations_to_use[i], :], color = colors[i], label = '$i$ = {:02.0f}'.format(inclinations_to_use[i]*5))

#now add in the MIMIR polarization data    
ax.loglog( [1.6, 2.1], [0.0056, 0.0067], marker = 'o', color = 'black')  # <-- data from JD = 2456676.8   [H=9.401, K=8.5184] 
ax.loglog( [1.6, 2.1], [0.0081, 0.0094], marker = 'o', color = 'black') # <-- data from JD = 2456736.8   [H=9.43, K=8.5106]

#put in a legend
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, bbox_to_anchor=(1.35, 1.25))

#plot limits, etc.
ax.set_xlim(0.3, 10.)
ax.set_ylim(2.e-5, 1)
ax.set_title(r'AA Tau Model')
ax.set_xlabel(r'$\lambda$ [$\mu$m]')
ax.set_ylabel(r'Polarization Fraction')
fig.savefig('Figures/AATau_linpol.png', bbox_inches='tight')

fig = plt.figure(figsize=(7, 4.5))
ax = fig.add_subplot(1, 1, 1)

#### plot the sources for all AA Tau's photons ####
# Total SED
seventy_degree_sed = AATau_mo.get_sed(inclination=15, aperture=-1, distance=137 * pc) #<--- using distance estimate corresponding to Gaia DR2 parallax
ax.loglog(seventy_degree_sed.wav, seventy_degree_sed.val, color='black', lw=3, alpha=0.5, label = 'total')

# Direct stellar photons
seventy_degree_direct_sed = AATau_mo.get_sed(inclination=15, aperture=-1, distance=137 * pc, component='source_emit')
ax.loglog(seventy_degree_direct_sed.wav, seventy_degree_direct_sed.val, color='blue', label = 'star - direct')

# Scattered stellar photons
seventy_degree_scat_sed = AATau_mo.get_sed(inclination=15, aperture=-1, distance=137 * pc, component='source_scat')
ax.loglog(seventy_degree_scat_sed.wav, seventy_degree_scat_sed.val, color='teal', label = 'star - scat')

# Direct dust photons
seventy_degree_dust_sed = AATau_mo.get_sed(inclination=15, aperture=-1, distance=137 * pc, component='dust_emit')
ax.loglog(seventy_degree_dust_sed.wav, seventy_degree_dust_sed.val, color='red', label = 'dust - emit')

# Scattered dust photons
seventy_degree_ScatDust_sed = AATau_mo.get_sed(inclination=15, aperture=-1, distance=137 * pc, component='dust_scat')
ax.loglog(seventy_degree_ScatDust_sed.wav, seventy_degree_ScatDust_sed.val, color='orange', label = 'dust - scat')


handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, bbox_to_anchor=(1.075,1.025))


ax.set_xlabel(r'$\lambda$ [$\mu$m]')
ax.set_ylabel(r'$\lambda F_\lambda$ [ergs/s/cm$^2$]')
ax.set_title(r'AA Tau Model (i ~ 75)')

ax.set_xlim(0.1, 2000.)
ax.set_ylim(2.e-16, 2.e-9)
fig.savefig('Figures/AATau_sed_plot_components_75.png')

##### Read in the AA Tau Spectra ######
#Read in AA Tau's 2008 IR spectrum, starting with the IR channel (using function Kevin made from Kristen's fancy code)
IRSpec_2008_wavelength, IRSpec_2008_flux = readSpectrum('spectra/AATau_IR2008.txt', u.micron)

AATau_NIReqws_2008, AATau_NIRlinefluxes_2008, AATau_lineIDs, AATau_lineCenters = measure_Covey2010lines(IRSpec_2008_wavelength, IRSpec_2008_flux)


#Now read in AA Tau's 2008 optical spectrum
OptSpec_2008_wavelength, OptSpec_2008_flux = readSpectrum('spectra/AATau_opt2008.txt', u.micron)

#Now read in AA Tau's 2014 IR spectra
IRSpec_dec2_2014_wavelength, IRSpec_dec2_2014_flux = readSpectrum('spectra/AATau_IR2014dec02.txt', u.micron)
IRSpec_dec12_2014_wavelength, IRSpec_dec12_2014_flux = readSpectrum('spectra/AATau_IR2014dec12.txt', u.micron)

AATau_NIReqws_2014, AATau_NIRlinefluxes_2014, AATau_lineIDs, AATau_lineCenters = measure_Covey2010lines(IRSpec_dec2_2014_wavelength, IRSpec_dec2_2014_flux)


#Now read in AA Tau's 2014 optical spectra
OptSpec_dec2_2014_wavelength, OptSpec_dec2_2014_flux = readSpectrum('spectra/AATau_opt2014dec02.txt', u.micron)
OptSpec_dec12_2014_wavelength, OptSpec_dec12_2014_flux = readSpectrum('spectra/AATau_opt2014dec12.txt', u.micron)

#Read in AA Tau's 2012 Xshooter spectra
IRSpec_2012_wavelength, IRSpec_2012_flux = readSpectrum('spectra/AA_Tau_2012_Xshooter.NIR.spec', u.micron)
optSpec_2012_wavelength, optSpec_2012_flux = readSpectrum('spectra/AA_Tau_2012_Xshooter.VIS.spec', u.micron)
UVSpec_2012_wavelength, UVSpec_2012_flux = readSpectrum('spectra/AA_Tau_2012_Xshooter.UVS.spec', u.micron)

#Read in AA Tau's 1998 NICMOS spectra
IRSpec_1998_wavelength, IRSpec_1998_flux = readSpectrum('spectra/AA_Tau_1998_NICMOS.spec', u.micron)

#Read in AA Tau's 2006 SpeX spectra
IRSpec_2006_wavelength, IRSpec_2006_flux = np.loadtxt('spectra/aatau.27nov06.txt', usecols=range(2), skiprows=117, unpack=True) 

AATau_NIReqws_2006, AATau_NIRlinefluxes_2006, AATau_lineIDs, AATau_lineCenters = measure_Covey2010lines(IRSpec_2006_wavelength, IRSpec_2006_flux)


fig = plt.figure(figsize = (5,4))
ax = fig.add_subplot(1,1,1)

plt.plot(AATau_lineCenters, 100.*(AATau_NIReqws_2006 - AATau_NIReqws_2014) / AATau_NIReqws_2006, marker = 'o', color = 'black', label = '2006 -> 2014')
plt.plot(AATau_lineCenters, 100.*(AATau_NIReqws_2008 - AATau_NIReqws_2014) / AATau_NIReqws_2008, marker = 'o', color = 'green', label = '2008 -> 2014')

handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, bbox_to_anchor=(1.2,0.2))

ax.set_xlabel(r'$\lambda$ [$\mu$m]')
ax.set_ylabel(r'$\Delta EqW (\%)$')
ax.set_title(r'(pre - post)/pre')

ax.set_xlim(1.45, 2.4)
#ax.set_ylim(2.e-16, 2.e-9)
fig.savefig('Figures/DeltaVeiling.png', bbox_inches='tight')

##### Now start measuring spectral features #####

temps = 30
gravities = 4
n_lines = 14

all_model_eqws = np.zeros( (temps, gravities, n_lines) )

all_temps = np.linspace(3000, 6000, num = 30)

for i in range(temps):
    this_temp = 3000+i*100
    string_temp = '0'+str(this_temp)
    for j in range(gravities):
        this_gravity = 1+j*1
        string_gravity = str(this_gravity)+'.00'
        #print(string_gravity)

        model_wave, model_flux, model_temp, model_hdulist = get_model(string_temp, string_gravity)
        model_eqws, model_linefluxes, modelID, model_center = measure_Covey2010lines(model_wave, model_flux)

        all_model_eqws[i, j, : ] = model_eqws

colors = [cmap(i) for i in np.linspace(0, 1, gravities)]
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14)  = plt.subplots(1, 14, sharey = True)

for i in range(n_lines):
    print('line '+str(i)+': ', AATau_NIReqws_2006[i], AATau_NIReqws_2008[i], AATau_NIReqws_2014[i]) 
    print('line '+str(i)+': ', AATau_NIReqws_2006[i]/AATau_NIReqws_2008[i], AATau_NIReqws_2008[i]/AATau_NIReqws_2008[i], AATau_NIReqws_2014[i]/AATau_NIReqws_2008[i]) 
    
    plt.subplot(1, 14, i+1)
    for j in range(gravities):
        plt.plot(all_model_eqws[:, j, i], all_temps, color = colors[j])
        #plt.setp(this_axis.get_xticklabels(), visible = False)
#fig.subplots_adjust(hspace=0)

    plt.plot([AATau_NIReqws_2008[i],AATau_NIReqws_2014[i],AATau_NIReqws_2006[i]], [4000,4000,4000], marker = 'o') 

plt.setp( [a.get_yticklabels() for a in fig.axes[1:14]], visible = False)
fig.savefig('Figures/SpT_indices.png', bbox_inches='tight')


#### plot the classification spaces from Covey2010 ####
plt.figsize=(14, 7)
fig, ( (ax1, ax2), (ax3, ax4) )  = plt.subplots(2, 2)

# first plot - blank

#second plot - CO / Ca vs. CO / Na + Ca
for j in range(gravities):
    ax2.plot( all_model_eqws[:, j, 13] / all_model_eqws[:, j, 9], all_model_eqws[:, j, 12] / ( all_model_eqws[:, j, 10] + all_model_eqws[:, j, 11]) ,color = colors[j])

ax2.plot( np.array([AATau_NIReqws_2006[13], AATau_NIReqws_2008[13], AATau_NIReqws_2014[13]]) / np.array( [AATau_NIReqws_2006[9], AATau_NIReqws_2008[9], AATau_NIReqws_2014[9]] ), np.array([AATau_NIReqws_2006[12], AATau_NIReqws_2008[12], AATau_NIReqws_2014[12]]) / ( np.array([AATau_NIReqws_2006[10], AATau_NIReqws_2008[10], AATau_NIReqws_2014[10]]) + np.array([AATau_NIReqws_2006[11], AATau_NIReqws_2008[11], AATau_NIReqws_2014[11]]) ), marker = 'o', color = 'black')
     
ax2.set_ylabel(r'CO 2.34 / Ca 1.98')
ax2.set_xlabel(r'CO 2.3 / <Na 2.21 + Ca 2.26>')
ax2.set_xlim(-1, 7)
ax2.set_ylim(-0.5, 3)

# third plot Mg/Si vs. Mg / CO
for j in range(gravities):
    ax3.plot( all_model_eqws[:, j, 2] / (all_model_eqws[:, j, 4] + all_model_eqws[:, j, 5]), all_model_eqws[:, j, 8] / all_model_eqws[:, j, 3], color = colors[j])

ax3.plot( np.array([AATau_NIReqws_2006[2], AATau_NIReqws_2008[2], AATau_NIReqws_2014[2]]) / ( np.array([AATau_NIReqws_2006[4], AATau_NIReqws_2008[4], AATau_NIReqws_2014[4]]) + np.array([AATau_NIReqws_2006[5], AATau_NIReqws_2008[5], AATau_NIReqws_2014[5]]) ), np.array([AATau_NIReqws_2006[8], AATau_NIReqws_2008[8], AATau_NIReqws_2014[8]]) / np.array( [AATau_NIReqws_2006[3], AATau_NIReqws_2008[3], AATau_NIReqws_2014[3]] ), marker = 'o', color = 'black')
     
ax3.set_ylabel(r'Mg 1.71 / Si 1.59')
ax3.set_xlabel(r'Mg 1.58 / CO 1.62 + 1.66')
ax3.set_xlim(-1, 5)
ax3.set_ylim(-1, 4)

# fourth plot Mg/Si vs. Mg / CO
for j in range(gravities):
    ax4.plot( all_model_eqws[:, j, 2] / (all_model_eqws[:, j, 4] + all_model_eqws[:, j, 5]), all_model_eqws[:, j, 8] / all_model_eqws[:, j, 3], color = colors[j])

ax4.plot( np.array([AATau_NIReqws_2006[2], AATau_NIReqws_2008[2], AATau_NIReqws_2014[2]]) / ( np.array([AATau_NIReqws_2006[4], AATau_NIReqws_2008[4], AATau_NIReqws_2014[4]]) + np.array([AATau_NIReqws_2006[5], AATau_NIReqws_2008[5], AATau_NIReqws_2014[5]]) ), np.array([AATau_NIReqws_2006[8], AATau_NIReqws_2008[8], AATau_NIReqws_2014[8]]) / np.array( [AATau_NIReqws_2006[3], AATau_NIReqws_2008[3], AATau_NIReqws_2014[3]] ), marker = 'o', color = 'black')
     
ax4.set_ylabel(r'KI 1.52 / Al 1.67')
ax4.set_xlabel(r'Mg 1.51 / 1.70 dip')
ax4.set_xlim(-1, 20)
ax4.set_ylim(-1, 2.5)

fig.savefig('Figures/SpT_classification_spaces.png', bbox_inches='tight')
    
#    figsize=(14, 7))
#ax = fig.add_subplot(14, 1, 1)







#try measuring EqWs for model spectra
#model_wave, model_flux, model_temp, model_hdulist = get_model('04000', '4.50')

#model_eqws, model_linefluxes = measure_Covey2010lines(model_wave, model_flux)

#print(model_eqws)

#import EqW measurement routine that Kristen stole from UVES tutorial and Kevin hacked to make sense to him.


