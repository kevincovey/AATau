import numpy as np

from hyperion.model import AnalyticalYSOModel
from hyperion.util.constants import rsun, lsun, au, msun, yr, c

# Initalize the model
m = AnalyticalYSOModel()

# Read in stellar spectrum
wav, fnu = np.loadtxt('kt04000g+3.5z-2.0.ascii', unpack=True)
nu = c / (wav * 1.e-4)

# Set the stellar parameters
m.star.radius = 1.85 * rsun
m.star.spectrum = (nu, fnu)
m.star.luminosity = lsun
m.star.mass = 0.8 * msun

# Add a flared disk
disk = m.add_flared_disk()
disk.mass = 0.03 * msun
disk.rmin = 7 * m.star.radius
disk.rmax = 400 * au    #from quote of Andrews & Williams
disk.r_0 = 0.08 * au #m.star.radius
disk.h_0 = 0.01 * disk.r_0
disk.p = -1.0
disk.beta = 1.25
disk.dust = 'kmh_lite.hdf5'

## # Add an Ulrich envelope
## envelope = m.add_ulrich_envelope()
## envelope.rc = disk.rmax
## envelope.mdot = 0 * msun / yr
## envelope.rmin = 7 * m.star.radius
## envelope.rmax = 500 * au
## envelope.dust = 'kmh_lite.hdf5'

## # Add a bipolar cavity
## cavity = envelope.add_bipolar_cavity()
## cavity.power = 1.5
## cavity.theta_0 = 90
## cavity.r_0 = envelope.rmax
## cavity.rho_0 = 5e3 * 3.32e-24
## cavity.rho_exp = 0.
## cavity.dust = 'kmh_lite.hdf5'

# Use raytracing to improve s/n of thermal/source emission
m.set_raytracing(True)

# Use the modified random walk
m.set_mrw(True, gamma=2.)

# Set up grid
m.set_spherical_polar_grid_auto(399, 199, 1)

# Set up SED
sed = m.add_peeled_images(sed=True, image=False)
n_view = 19
sed.set_viewing_angles(np.linspace(0., 90., n_view), np.repeat(45., n_view))
sed.set_wavelength_range(150, 0.02, 2000.)
sed.set_stokes(True)
sed.set_track_origin('detailed')

# Set number of photons
m.set_n_photons(initial=1e6, imaging=1e6,
                raytracing_sources=1e4, raytracing_dust=1e6)

# Set number of temperature iterations and convergence criterion
m.set_n_initial_iterations(10)
m.set_convergence(True, percentile=99.0, absolute=2.0, relative=1.1)

# Write out file
m.write('AATau_example.rtin')
m.run('AATau_example.rtout', mpi=True)
