import numpy as np
import matplotlib.pyplot as plt
from pystaff import SpectralFitting_functs as SF
import pickle
import os
import corner
import pandas as pd
from astropy.table import Table
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

# *********************************************************************************************** #
#                                    Analyse the Pystaff outputs
# *********************************************************************************************** #
plt.ion()
date = datetime.today().strftime("%d-%m-%y")

# ----------------------------------------------------------------------------------------------- #
#                                   Set up directories
# ----------------------------------------------------------------------------------------------- #
# define the directories
# File and path names to set:
name = "red_eyebrow"
nwalkers = 200

if name == "red_eyebrow":
    acronym = "REB"
elif name == "rosetta_stones":
    acronym = "RS"

filename_format = "REB_jwst_steps10000"
results_dir = f"output_results/{name}/{filename_format}/"

# ----------------------------------------------------------------------------------------------- #
#                                            Load data
# ----------------------------------------------------------------------------------------------- #
# load the fit_obj object
with open(results_dir + "fit_obj.pckl", "rb") as file:
    fit_obj = pickle.load(file)

# load the results_theta.pckl (this contains the LM fit parameters)
with open(results_dir + "results_theta.pckl", "rb") as file:
    results_theta = pickle.load(file)

# load the MCMC chains
samples = np.genfromtxt(results_dir + "mpi_samples.txt")

output_dir = results_dir + "figure_outputs/"
# check whether the output) directory exists, if not create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# get bestfit
likelihood, [wave, bestfit, noise, flux, sky, emission_lines, polys, weights] = SF.lnlike(
    results_theta, fit_obj.fit_settings, ret_specs=True
)

# express the residuals as a percentage offset
residuals = 100 * (flux - bestfit) / flux

# get the number of steps based on the number of walkers
nsteps = samples.shape[0] // nwalkers
nparams = samples.shape[1]

# samples_tbl = Table(data=samples, names=results_theta.keys())

# This was a fit_type=1 fit (simple model), so these are the params of interest
params_of_interest = np.array(["logage", "zH", "a", "C", "N", "Mg", "Si", "Ca", "Ti", "Na"])
params_of_interest2 = np.array(["chi2", "velz", "sigma"])
params_of_interest3 = np.array(["IMF1", "IMF2", "logage", "zH", "a"])
params_of_interest4 = np.array(["IMF1", "IMF2", "ML_v", "ML_i", "MW_v", "MW_i"])
params_of_interest5 = np.array(
    ["IMF1", "IMF2", "logage", "logfy", "fy logage", "logm7g", "loghot", "hotteff"]
)
# note the 'a' refers to the alpha element (predominantely Oxygen), not the IMF slope

# ----------------------------------------------------------------------------------------------- #
#                         Use Pystaff's inbuilt fit funcs and corner.corner
# ----------------------------------------------------------------------------------------------- #
# plot the fit using the inbuilt plotting function
SF.plot_fit(results_theta, fit_obj.fit_settings)
plt.savefig(f"{output_dir}pystaff_bestfit.pdf", bbox_inches="tight")

# Plot a corner plot of all the variables
corner.corner(samples, labels=list(results_theta.keys()))
plt.savefig(f"{output_dir}corner_plot.pdf")

# ----------------------------------------------------------------------------------------------- #
#                                     Plot the fit
# ----------------------------------------------------------------------------------------------- #

fig0, ax0 = plt.subplots(figsize=(12, 5), nrows=2)
# plot the data, model and uncertainty
ax0[0].fill_between(
    x=wave,
    y1=flux + noise,
    y2=flux - noise,
    color="grey",
    alpha=0.1,
    step="pre",
    label="Uncertainty",
)
ax0[0].plot(wave, flux, c="k", ds="steps", lw=0.5, label="Data")
ax0[0].plot(wave, bestfit, c="orange", ds="steps", lw=0.5, label="Model")

# plot the residuals
ax0[1].plot(wave, np.zeros_like(wave), c="k", lw=0.5, ds="steps", alpha=0.5)
ax0[1].plot(wave, residuals, c="green", ds="steps", lw=0.5)
ax0[1].fill_between(x=wave, y1=noise * 100, y2=-noise * 100, color="grey", alpha=0.1, step="pre")
# setup the axes
ax0[1].set_xlabel(r"Rest-Frame Wavelength [${\AA}$]", fontsize=15)
ax0[0].set_ylabel("Flux (Normalised)", fontsize=15)
ax0[1].set_ylabel("Residuals (%)", fontsize=15)
# set the ranges

std_d_flux = np.std(flux)
std_resid = np.std(residuals)
xlim = [wave.min(), wave.max()]
ylim0 = [flux.min() - std_d_flux * 0.1, flux.max() + std_d_flux * 0.1]
ylim1 = [-residuals.min() - std_resid * 0.1, -residuals.max() + std_resid * 0.1]

ax0[0].set_xlim(xlim)
ax0[1].set_xlim(xlim)
ax0[0].set_ylim(ylim0)
ax0[1].set_ylim(ylim1)

# plot IMF sensitive features
# IMF sensitive features from Conroy and van Dokkum 2012a table 1
conroy2012 = Table.from_pandas(
    pd.read_excel("tbar_files/Conroy2012a_table1.xlsx", sheet_name="reformatted")
)

# plot the IMF senstive spectral features listed in Conroy and van Dokkum 2012a
for row in conroy2012:
    line_left = row["Feature left"]
    line_right = row["Feature right"]
    line_centre = line_left + 0.5 * (line_right - line_left)
    text_y = 0.9

    for i in range(2):
        ax0[i].axvspan(xmin=line_left, xmax=line_right, alpha=0.1, color="xkcd:blue")
        if i == 0:  # only plot the text on the top panel
            ax0[i].annotate(
                text=row["Index"],
                xy=(line_centre, text_y),
                xytext=(line_centre, text_y),
                rotation="vertical",
            )

# save the figure
fig0.savefig(f"{output_dir}{name}_{date}_model_spectrum.pdf", bbox_inches="tight")

# ----------------------------------------------------------------------------------------------- #
#                             Plot the MCMC chains
# ----------------------------------------------------------------------------------------------- #

outname = f"{output_dir}{name}_{date}_traces_nwalks{int(nwalkers)}" f"_nchain{int(nsteps)}.pdf"

with PdfPages(outname) as pdf:
    for i, label in enumerate(dict(results_theta)):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_ylabel(label, fontsize=20)
        ax.set_xlabel("Steps", fontsize=20)
        for w in range(nwalkers):
            ax.plot(
                samples_tbl[label][w * nsteps : (w + 1) * nsteps], color="k", alpha=0.1, ds="steps"
            )
        pdf.savefig()
        plt.close(fig)

"""
# --------------------------------------------------------------------------- #
#                   Have a closer look at the Chi2 chains
# --------------------------------------------------------------------------- #
# where 53 is the number of parameters
# ignore the "burnin" period of 1000 steps
mcmc_reshaped = obj.mcmc.reshape(obj.Nchain, obj.Nwalkers, 53)
fig1, ax1 = plt.subplots(figsize=(12, 5))
for chain in range(obj.Nwalkers):
    # plot the chi2 for each walker
    ax1.plot(mcmc_reshaped[1000:, chain, 0], lw=0.05, alpha=0.05, c="k")

# --------------------------------------------------------------------------- #
#                             Make corner plots
# --------------------------------------------------------------------------- #
plt.close("all")
obj.plot_corner(
    outname=f"{output_dir}{name}_stellar_pops_corner_{date}"
    f"_nwalks{int(obj.Nwalkers)}_nchain{int(obj.Nchain)}.pdf",
    params=params_of_interest,
    figsize=(25, 25),
    burnin=Nburn,
)

plt.close("all")
obj.plot_corner(
    outname=f"{output_dir}{name}_LOSVD_corner_{date}"
    f"_nwalks{int(obj.Nwalkers)}_nchain{int(obj.Nchain)}.pdf",
    params=params_of_interest2,
    figsize=(20, 20),
    burnin=Nburn,
)

plt.close("all")
obj.plot_corner(
    outname=f"{output_dir}{name}_imf_params_{date}"
    f"_nwalks{int(obj.Nwalkers)}_nchain{int(obj.Nchain)}.pdf",
    params=params_of_interest3,
    figsize=(20, 20),
    burnin=Nburn,
)

plt.close("all")
obj.plot_corner(
    outname=f"{output_dir}{name}_MLparams_{date}"
    f"_nwalks{int(obj.Nwalkers)}_nchain{int(obj.Nchain)}.pdf",
    params=params_of_interest4,
    figsize=(20, 20),
    burnin=Nburn,
)

plt.close("all")
obj.plot_corner(
    outname=f"{output_dir}{name}_age_parameters_{date}"
    f"_nwalks{int(obj.Nwalkers)}_nchain{int(obj.Nchain)}.pdf",
    params=params_of_interest5,
    figsize=(20, 20),
    burnin=Nburn,
)

# --------------------------------------------------------------------------- #
#                  Look at the IMF mismatch parameter
# --------------------------------------------------------------------------- #
plt.close("all")

obj.plot_corner(
    outname=f"{output_dir}{name}_iband_ML_{date}"
    f"_nwalks{int(obj.Nwalkers)}_nchain{int(obj.Nchain)}.pdf",
    params=["ML_i", "MW_i"],
    figsize=(20, 20),
    burnin=Nburn,
)

obj.plot_corner(
    outname=f"{output_dir}{name}_vband_ML_{date}"
    f"_nwalks{int(obj.Nwalkers)}_nchain{int(obj.Nchain)}.pdf",
    params=["ML_v", "MW_v"],
    figsize=(20, 20),
    burnin=Nburn,
)

bands = ["v", "i"]
alpha = [obj.get_chains_for_param(f"ML_{b}") / obj.get_chains_for_param(f"MW_{b}") for b in bands]

colors = ["xkcd:sky blue", "xkcd:light red"]
fig, ax = plt.subplots(ncols=1)
for i in range(2):
    ax.hist(alpha[i], bins=500, alpha=0.5, color=colors[i], label=bands[i])

ax.set_title(r"IMF mismatch parameter $\alpha = ML / ML(MW)$", fontsize=20)
ax.legend()
fig.savefig(
    f"{output_dir}{name}_{date}_IMF_mismatch_parameter.pdf",
    bbox_inches="tight",
)

# Rather than plotting a histogram of alpha, plot it as a function of mcmc step so can see
# how it changes with time.
aa = alpha[0].reshape([obj.Nchain, obj.Nwalkers])  # just take one band for now
fig, ax = plt.subplots()
ax.set_ylabel(r"IMF mismatch parameter $\alpha$")
for i in range(obj.Nwalkers):
    ax.plot(np.arange(obj.Nchain), aa[:, i], lw=0.1, c="k")
fig.savefig(f"{output_dir}{name}_IMF_mismatch_parameter_vs_steps.pdf", bbox_inches="tight")

# save the object to file!
with open(results_dir + filename_format + "_obj.pckl", "wb") as file:
    pickle.dump(obj, file)
"""
