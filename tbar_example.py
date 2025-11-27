#! /usr/bin/env python
##############################################################################
import numpy as np
import emcee
import scipy.interpolate as si
import lmfit as LM
import matplotlib.pyplot as plt
import corner
import pickle
import os
import argparse

from pystaff.SpectralFitting import SpectralFit
from pystaff import SpectralFitting_functs as SF

from schwimmbad import MPIPool

# Get the number of steps, the input data file, and runname parsed from command-line arguments
parser = argparse.ArgumentParser(description="Run spectral fitting with MCMC")
parser.add_argument(
    "--target",
    type=str,
    default=None,
    help='Target name, either "red_eyebrow" or "rosetta_stones"',
)
parser.add_argument(
    "--filename",
    type=str,
    default="data/REB_jwst.dat",
    help="Input data file path (default: data/REB_jwst.dat)",
)
parser.add_argument(
    "--nsteps", type=int, default=100, help="Number of MCMC steps (default is 100)"
)

parser.add_argument(
    "--run-name",
    type=str,
    default=None,
    help="Name for this run (default: auto-generated from nsteps)",
)
parser.add_argument(
    "--nwalkers", type=int, default=70, help="Number of MCMC walkers (default: 70)"
)

args = parser.parse_args()

# Generate run_name if not provided
if args.run_name is None:
    run_name = f"{args.filename.split('/')[-1].replace('.dat', '')}_steps{args.nsteps}"
else:
    run_name = args.run_name

nsteps = args.nsteps
nwalkers = args.nwalkers
datafile = args.filename

# make the ouput directory if it doesn't already exist
output_dir = f"output_results/{args.target}/{run_name}/"
os.makedirs(output_dir, exist_ok=True)


# Likelihood function here. We could put it in the SpectraFitting class, but when
# working with MPI on a cluster that would mean we'd need to pickle the fit_settings
# dictionary, which massively slows things down
def lnprob(T, theta, var_names, bounds, ret_specs=False):

    # Log prob function. T is an array of values

    assert len(T) == len(
        var_names
    ), "Error! The number of variables and walker position shapes are different"

    # Prior information comes from the parameter bounds now
    if np.any(T > bounds[:, 1]) or np.any(T < bounds[:, 0]):
        return -np.inf

    # make theta from the emcee walker positions
    for name, val in zip(var_names, T):
        theta[name].value = val

    if ret_specs is False:
        ll = SF.lnlike(theta, fit.fit_settings)
        return ll
    else:
        return SF.lnlike(theta, fit.fit_settings, ret_specs=True)


# Can select either Kroupa or Salpeter to use with the SSP models
element_imf = "kroupa"

####################################################

# Load the data
# The instrumental resolution can be included if it's known. We need a value of sigma_inst in km/s
# for every pixel. Otherwise put it as None
lamdas_orig, flux_orig, errors, pixel_weights, instrumental_resolution_orig = np.genfromtxt(
    datafile, unpack=True
)

# need to make the wavelength scale slightly more consistent.
lamdas = np.arange(lamdas_orig[0], lamdas_orig[-1], step=lamdas_orig[1] - lamdas_orig[0])
# reinterpolate the flux
interper1 = si.interp1d(x=lamdas_orig, y=flux_orig)
flux = interper1(lamdas)
# Also need to reinterpret the instrumental resolution
interper2 = si.interp1d(x=lamdas_orig, y=instrumental_resolution_orig)
instrumental_resolution = interper2(lamdas)
instrumental_resolution = None

# TEST: shift the wavelength array to match the example one
# lamdas -= 2230

fig, ax = plt.subplots()
ax.plot(lamdas_orig, flux_orig, lw=0.5, label="Original wavelength grid")
ax.plot(lamdas, flux, lw=0.5, label="Interpolated flux/wavelength")
ax.legend()

# Sky Spectra
# Give a list of 1D sky spectra to be scaled and subtracted during the fit
# Otherwise leave sky as None
skyspecs = None
# ######################################################


# ######################################################
# Mask out regions that we don't want to fit, e.g. around telluric residuals, particularly nasty skylines, etc
# THESE SHOULD BE OBSERVED WAVELENGTHS
# A few examples of areas I often avoid due to skylines or telluric residuals
telluric_lam_1 = np.array([[6862, 6952]])
telluric_lam_2 = np.array([[7586, 7694]])
skylines = np.array([[8819, 8834], [8878.0, 8893], [8911, 8925], [8948, 8961]])

masked_wavelengths = np.vstack([telluric_lam_1, telluric_lam_2, skylines]).reshape(-1, 1, 2)
string_masked_wavelengths = [
    "{} to {}".format(pair[0][0], pair[0][1]) for pair in masked_wavelengths
]

# Make a mask of pixels we don't want
pixel_mask = np.ones_like(flux, dtype=bool)
for array in masked_wavelengths:
    m = SF.make_mask(lamdas, array)
    pixel_mask = m & pixel_mask

# Now switch the weights of these pixels to 0
pixel_weights = np.ones_like(flux)
pixel_weights[~pixel_mask] = 0.0


# Wavelengths we'll fit between.
# Split into 4 to make the multiplicative polynomials faster
# fit_wavelengths = np.array([[8000, 9200]])
fit_wavelengths = np.array(
    [[7004, 8240], [8240, 9600], [9600, 10700], [10700, 12200], [12200, 13368]]
)
string_fit_wavelengths = ["{} to {}".format(pair[0], pair[1]) for pair in fit_wavelengths]

# FWHM.
# This should be the FWHM in pixels of the instrument used to observe the spectrum.
FWHM_gal = 3.0

# Now set up the spectral fitting class
print("Setting up the fit")
# These should be the location of the folder containing all the templates
# The code will use 'glob' to search for all templates matching the correct filenames.

# tbar's project dir
# proj_dir = "/Users/tbarone/Desktop/Swinburne/AGEL/InitialMassFunction/stellar_pops_fitting/"
# base_template_location = proj_dir + "alf_infiles/infiles/"
# varelem_template_location = proj_dir + "alf_infiles/infiles/"

ALF_proj_dir = "/fred/oz041/tbarone/softwares/alf_rosetta_stones/"
base_template_location = ALF_proj_dir + "/infiles/"
varelem_template_location = ALF_proj_dir + "/infiles/"

fit = SpectralFit(
    lamdas,
    flux,
    errors,
    pixel_weights,
    fit_wavelengths,
    FWHM_gal,
    instrumental_resolution=instrumental_resolution,
    skyspecs=skyspecs,
    element_imf=element_imf,
    base_template_location=base_template_location,
    varelem_template_location=varelem_template_location,
)
fit.set_up_fit()


# ------------------------------------------------------------------------------------
# Wrap all the parameter setup + MCMC in a function so we can pass in an MPI pool
# ------------------------------------------------------------------------------------
def main(pool, nsteps=10, nwalkers=70):
    global fit  # used inside lnprob

    # Here are the available fit parameters
    # They can easily be switched off by changing vary to False
    # The min and max values act as flat priors
    theta = LM.Parameters()
    # LOSVD parameters
    theta.add("Vel", value=0, min=-1000.0, max=10000.0)
    theta.add("sigma", value=210.0, min=10.0, max=700.0)

    # Abundance of Na. Treat this separately, since it can vary up to +1.0 dex
    theta.add("Na", value=0.5, min=-0.45, max=1.0, vary=True)

    # Abundance of Carbon. Treat this separately, since its templates are at +/- 0.15 dex rather than +/- 0.3
    theta.add("C", value=0.0, min=-0.2, max=0.2, vary=True)

    # Abundance of elements which can vary positively and negatively
    theta.add("Ca", value=0.0, min=-0.45, max=0.45, vary=True)
    theta.add("Fe", value=0.0, min=-0.45, max=0.45, vary=True)
    theta.add("N", value=0.0, min=-0.45, max=0.45, vary=True)
    theta.add("Ti", value=0.0, min=-0.45, max=0.45, vary=True)
    theta.add("Mg", value=0.0, min=-0.45, max=0.45, vary=True)
    theta.add("Si", value=0.0, min=-0.45, max=0.45, vary=True)
    theta.add("Ba", value=0.0, min=-0.45, max=0.45, vary=True)

    # Abundance of elements which can only vary above 0.0
    theta.add("as_Fe", value=0.0, min=0.0, max=0.45, vary=True)
    theta.add("Cr", value=0.0, min=0.0, max=0.45, vary=True)
    theta.add("Mn", value=0.0, min=0.0, max=0.45, vary=True)
    theta.add("Ni", value=0.0, min=0.0, max=0.45, vary=True)
    theta.add("Co", value=0.0, min=0.0, max=0.45, vary=True)
    theta.add("Eu", value=0.0, min=0.0, max=0.45, vary=True)
    theta.add("Sr", value=0.0, min=0.0, max=0.45, vary=True)
    theta.add("K", value=0.0, min=0.0, max=0.45, vary=True)
    theta.add("V", value=0.0, min=0.0, max=0.45, vary=True)
    theta.add("Cu", value=0.0, min=0.0, max=0.45, vary=True)

    # Base population parameters
    # Age, Metallicity, and the two IMF slopes
    theta.add("age", value=3.1, min=1.0, max=14.0)
    theta.add("Z", value=-0.2, min=-1.0, max=0.2)
    theta.add("imf_x1", value=2.35, min=0.5, max=3.5)
    theta.add("imf_x2", value=2.35, min=0.5, max=3.5)

    # Option to rescale the error bars up or down
    theta.add("ln_f", value=0.0, min=-5.0, max=5.0, vary=True)

    # Select the parameters we're varying, ignore the fixed ones
    variables = [thing for thing in theta if theta[thing].vary]
    ndim = len(variables)
    # Vice versa, plus add in the fixed value
    fixed = [
        "{}={},".format(thing, theta[thing].value) for thing in theta if not theta[thing].vary
    ]
    print("fixed variables are: ", fixed)
    print("variable variables are: ", variables)

    # Added by TBAR
    fit.fit_settings["emission_lines"] = None

    # Optionally plot the fit with our initial guesses
    SF.plot_fit(theta, fit.fit_settings)

    ###############################################################################################
    # Set up the initial positions of the walkers as a ball with a different standard deviation in
    # each dimension
    nwalkers = nwalkers
    nsteps = nsteps

    # Get the spread of the starting positions
    stds = []
    n_general = 9
    n_positive = 10

    # Add in all these standard deviations
    # Kinematic parameters
    stds.extend([100.0, 25.0])
    # General parameters
    stds.extend([0.1] * n_general)
    # Positive parameters
    stds.extend([0.1] * n_positive)

    # Age
    stds.extend([1.0])

    # Z, imf1, imf2
    stds.extend([0.1, 0.1, 0.1])

    # ln_f
    stds.extend([0.5])

    stds = np.array(stds)

    assert len(stds) == len(
        variables
    ), "You must have the same number of dimensions for the Gaussian ball as variables!"

    # Now get the starting values for each parameter, as well as the prior bounds
    start_values, bounds = SF.get_start_vals_and_bounds(theta)
    p0 = SF.get_starting_positions_for_walkers(start_values, stds, nwalkers, bounds)
    ###################################################################################################
    # Do the sampling
    # This may take a while!

    # ###################################################################################################

    print("Running the fitting with {} walkers for {} steps".format(nwalkers, nsteps))
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, args=[theta, variables, bounds], pool=pool
    )
    result = sampler.run_mcmc(p0, nsteps)

    ####################################################################################################

    chain = sampler.get_chain()

    # get rid of the burn-in
    burnin = np.array(nsteps - 200).clip(0)
    samples = chain[burnin:, :, :].reshape((-1, ndim))
    print("\tDone")

    # Get the 16th, 50th and 84th percentiles of the marginalised posteriors for each parameter
    best_results = np.array(
        list(
            map(
                lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                zip(*np.percentile(samples, [16, 50, 84], axis=0)),
            )
        )
    )
    # If your posterior surface isn't a nice symmetric Gaussian, then the vector of median values for each parameter (as we're doing here)
    # could very well correspond to an unlikely area of parameter space and you'll need to do something different to this!

    for v, r in zip(variables, best_results):
        print("{}: {:.3f} +{:.2f}/-{:.2f}".format(v, r[0], r[1], r[2]))

    # Make a set of parameters with the results
    results_theta = LM.Parameters()
    for v, r in zip(variables, best_results):
        print(v, r)
        results_theta.add("{}".format(v), value=r[0], vary=True)
    # and include the things we kept fixed originally too:
    [
        results_theta.add("{}".format(thing), value=theta[thing].value, vary=False)
        for thing in theta
        if not theta[thing].vary
    ]

    # ... and plot
    SF.plot_fit(results_theta, fit.fit_settings)

    # Save samples
    np.savetxt(f"{output_dir}mpi_samples.txt", samples)

    # and save the results_theta object
    with open(f"{output_dir}results_theta.pckl", "wb") as file:
        pickle.dump(results_theta, file)


# ------------------------------------------------------------------------------------
# MPI entry point with schwimmbad.MPIPool
# ------------------------------------------------------------------------------------

# Choose how many worker processes you want – e.g. number of physical cores
nproc = int(os.environ.get("SLURM_NTASKS", 4))  # defaults to 4 if not in SLURM

# print(f"Using MPIPool with {nproc} processes")
# with MultiPool(processes=nproc) as pool:
#    main(pool=pool, nsteps=nsteps, nwalkers=nwalkers)

# Use MPI for cluster parallelization
# with MPIPool() as pool:
#    if not pool.is_master():
#        pool.wait()
#        sys.exit(0)
#
#    print(f"Running with {pool.size} MPI processes")
#

main(pool=None, nsteps=nsteps, nwalkers=nwalkers)

# Save the fit figure
plt.savefig(f"{output_dir}fit_results.pdf")

# and save the object as a pickle
with open(f"{output_dir}fit_obj.pckl", "wb") as file:
    pickle.dump(fit, file)

####################################################################################################

# It's always a good idea to inspect the traces
# Can also make corner plots, if you have corner available:

# reload the saved mcmc chains and results_theta
# samples = np.genfromtxt(f"{output_dir}mpi_samples.txt")

# with open(f"{output_dir}restults_theta.pckl", "rb") as file:
#    results_theta = pickle.load(file)

# corner.corner(samples, labels=list(results_theta.keys()))
# plt.savefig(f"{output_dir}corner_plot.pdf")

# And you should inspect the residuals around the best fit as a function of wavelength

###################################################################################################
