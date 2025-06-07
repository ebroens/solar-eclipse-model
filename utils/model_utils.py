# This file is part of Solar Eclipse Model.
# Copyright (C) 2025 Eric Broens
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


"""
model_utils.py

Helper functions for MCMC analysis, plotting, and model parameter handling.
Used by model_eclipse_mcmc.py to support sampling and result visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import csv
import os
from astropy.time import Time
import corner
from custom_models.solareclipsemodel import SolarEclipseModel


def run_sampler(obs_flux, sigma_data, fit_config, params, nwalkers, nsteps, output_dir, save_data=False):
    """
    Run the MCMC sampler using the `emcee` library.

    This function defines the log-prior, log-likelihood, and log-posterior
    functions, and then uses `emcee.EnsembleSampler` to explore the parameter
    space. It also handles an optional jitter term (`log_sigma_jitter`)
    to account for additional unknown uncertainties in the data.

    Parameters
    ----------
    obs_flux : np.ndarray
        Array of normalized observed flux values.
    sigma_data : float
        The estimated measurement uncertainty (standard deviation) of the flux data.
    fit_config : dict
        Dictionary specifying the fit parameters, their prior ranges,
        and setter functions. This dict is built by `build_fit_config`.
    params : SolarEclipseParams
        An instance of `SolarEclipseParams` representing the current state
        of the model's physical parameters. This object is modified
        in-place by the `setter` functions during sampling.
    nwalkers : int
        The number of walkers to use in the `emcee` ensemble sampler.
    nsteps : int
        The number of MCMC steps (iterations) to run the sampler.
    output_dir : str
        The directory where sampler state (if `save_data` is True) will be saved.
    save_data : bool, optional
        If True, the `emcee` sampler object will be pickled and saved to
        `output_dir`. Defaults to False.

    Returns
    -------
    tuple
        - samples : np.ndarray
            Flattened MCMC samples after discarding the burn-in period.
            Shape is (n_samples * (nsteps - burn_in), ndim).
        - log_probs : np.ndarray
            Log-probability values corresponding to the `samples`.
    """
    import emcee
    import dill

    labels = list(fit_config.keys()) # Get the order of parameters for theta

    def log_prior(theta):
        """
        Calculates the log-prior probability for a given set of parameters.

        This function enforces uniform priors specified in `fit_config`.
        If any parameter falls outside its defined prior range, a log-prior of
        -inf is returned, effectively rejecting that parameter combination.

        Parameters
        ----------
        theta : array_like
            An array of parameter values for which to calculate the log-prior.

        Returns
        -------
        float
            The log-prior probability. Returns -np.inf if any parameter is
            outside its prior bounds, otherwise returns 0.0 (for uniform priors).
        """
        lp = 0.0
        for i, label in enumerate(labels):
            low, high = fit_config[label]["prior"]
            if not (low < theta[i] < high):
                return -np.inf
        return 0.0

    def log_likelihood(theta, obs_flux, sigma_data):
        """
        Calculates the log-likelihood of the observed data given the model parameters.

        This function updates the global `params` object with the current `theta`
        values, generates the model light curve, and computes the Gaussian
        log-likelihood. It includes handling for an additional 'jitter' term
        (`log_sigma_jitter`) that adds in quadrature to the observational uncertainty.

        Parameters
        ----------
        theta : array_like
            An array of current parameter values from the MCMC sampler.
        obs_flux : np.ndarray
            Array of normalized observed flux values.
        sigma_data : float
            The estimated measurement uncertainty (standard deviation) of the flux data.

        Returns
        -------
        float
            The log-likelihood probability. Returns -np.inf if model evaluation fails.
        """
        n_params = len(fit_config)

        # Apply current theta values to the SolarEclipseParams object
        for i, label in enumerate(labels):
            fit_config[label]["setter"](params, theta[i])

        try:
            # Generate the model light curve with the current parameters
            model_flux = model.light_curve(params)
        except Exception as e:
            # If model evaluation fails, return -inf to reject this step
            print(f"Model evaluation error: {e}")
            return -np.inf

        sigma_eff = sigma_data
        # If 'log_sigma_jitter' is a fitted parameter, calculate effective uncertainty
        if "log_sigma_jitter" in fit_config.keys():
            log_sigma_jitter = theta[n_params-1]  # The last parameter is assumed to be log(sigma_jitter)
            sigma_jitter = np.exp(log_sigma_jitter)
            if sigma_data is None: # If no input sigma_data, use jitter as the only uncertainty
                sigma_eff = sigma_jitter
            else: # Combine data uncertainty and jitter in quadrature
                sigma_eff = np.sqrt(sigma_data**2 + sigma_jitter**2)

        # Full Gaussian log-likelihood: includes normalization to penalize small sigma_jitter
        # This prevents the sampler from favoring overly small uncertainties (overconfident fits)
        return -0.5 * np.sum(((obs_flux - model_flux) / sigma_eff) ** 2 + np.log(2 * np.pi * sigma_eff**2))

    def log_posterior(theta, obs_flux, sigma_data):
        """
        Calculates the log-posterior probability (log-prior + log-likelihood).

        Parameters
        ----------
        theta : array_like
            An array of current parameter values from the MCMC sampler.
        obs_flux : np.ndarray
            Array of normalized observed flux values.
        sigma_data : float
            The estimated measurement uncertainty (standard deviation) of the flux data.

        Returns
        -------
        float
            The log-posterior probability. Returns -np.inf if the log-prior is -inf.
        """
        lp = log_prior(theta)
        return lp + log_likelihood(theta, obs_flux, sigma_data) if np.isfinite(lp) else -np.inf


    ndim = len(labels) # Number of dimensions (parameters)
    # Initialize walker positions by sampling uniformly from the prior range for each parameter
    p0 = np.array([np.random.uniform(*fit_config[label]["prior"], nwalkers) for label in labels]).T

    # Initialize a SolarEclipseModel instance (used within the log_likelihood)
    # The num_entries argument is a dummy here as the time array comes from params.obs_datetime
    model = SolarEclipseModel(params, 1)

    # Initialize the emcee sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(obs_flux, sigma_data))
    # Run the MCMC sampler
    sampler.run_mcmc(p0, nsteps, progress=True)

    burn_in = 300 # Number of initial steps to discard to account for chain warm-up
    samples = sampler.get_chain(discard=burn_in, flat=True) # Flattened samples after burn-in
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True) # Corresponding log-probabilities

    if save_data:
        # Save the full sampler object for later inspection/resumption
        sampler_filepath = os.path.join(output_dir, f"sampler_{params.model_name}.pkl")
        with open(sampler_filepath, "wb") as f:
            dill.dump(sampler, f)
        print(f"Sampler object saved to: {sampler_filepath}")

    return samples, log_probs


def get_fit_summary(samples, log_probs, labels):
    """
    Summarize MCMC parameter statistics from the posterior samples.

    Calculates the best-fit parameters (corresponding to the maximum log-probability),
    the median of the posterior distributions, and the 16th and 84th percentiles
    to define the 1-sigma credible intervals.

    Parameters
    ----------
    samples : np.ndarray
        Flattened MCMC posterior samples for all parameters.
        Shape is (n_samples, ndim).
    log_probs : np.ndarray
        Log-probability values corresponding to each sample.
    labels : list of str
        A list of parameter names, in the same order as in `samples`.

    Returns
    -------
    dict
        A dictionary where keys are parameter labels and values are dictionaries
        containing:
        - "best_fit" (float): Parameter value at the maximum log-probability.
        - "median" (float): Median of the posterior distribution.
        - "minus_1sigma" (float): Difference between median and 16th percentile.
        - "plus_1sigma" (float): Difference between 84th percentile and median.
    """
    results = {}
    best_idx = np.argmax(log_probs) # Index of the sample with the highest log-probability
    best_params = samples[best_idx] # Parameter values at the best_idx

    for i, label in enumerate(labels):
        # Calculate 16th, 50th (median), and 84th percentiles for each parameter
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        # Calculate the 1-sigma lower and upper uncertainties
        q = np.diff(mcmc) # q[0] = median - 16th_percentile, q[1] = 84th_percentile - median
        results[label] = {
            "best_fit": best_params[i],
            "median": mcmc[1],
            "minus_1sigma": q[0],
            "plus_1sigma": q[1],
        }
    return results


def save_results_to_csv(results, filename):
    """
    Save the MCMC fit summary to a CSV file.

    Parameters
    ----------
    results : dict
        A dictionary containing the parameter summary statistics, as returned
        by `get_fit_summary`.
    filename : str
        The full path and filename for the output CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Best Fit", "Median", "Minus 1 Sigma", "Plus 1 Sigma"])
        for label, r in results.items():
            writer.writerow([
                label,
                f"{r['best_fit']:.6f}",    # Format to 6 decimal places
                f"{r['median']:.6f}",
                f"{r['minus_1sigma']:.6f}",
                f"{r['plus_1sigma']:.6f}"
            ])
    print(f"Fit results saved to: {filename}")


def update_params_from_results(params, results, fit_config):
    """
    Update a `SolarEclipseParams` object with median values from MCMC posterior summary.

    This function iterates through the `results` dictionary (from `get_fit_summary`)
    and uses the `setter` functions defined in `fit_config` to apply the median
    posterior values to the `params` object.

    Parameters
    ----------
    params : SolarEclipseParams
        The `SolarEclipseParams` object to be updated. This object is modified in-place.
    results : dict
        A dictionary containing the parameter summary statistics, as returned
        by `get_fit_summary`.
    fit_config : dict
        The fit configuration dictionary, which contains the `setter` functions
        for each parameter.

    Returns
    -------
    SolarEclipseParams
        The updated `SolarEclipseParams` object.
    """
    for label, value in results.items():
        # Use the setter function from fit_config to update the parameter with its median value
        if label in fit_config: # Ensure the label exists in fit_config
            fit_config[label]["setter"](params, value["median"])
    return params


def get_uncertainty_interval(n_samples, samples, fit_config, labels, params_template):
    """
    Calculate the 1-sigma uncertainty interval for the model light curve.

    This function randomly samples `n_samples` from the MCMC posterior chains,
    evaluates the model light curve for each sampled parameter set, and then
    computes the 16th and 84th percentiles across these model realizations
    to define the 1-sigma uncertainty band.

    Parameters
    ----------
    n_samples : int
        The number of MCMC samples to use for calculating the uncertainty interval.
    samples : np.ndarray
        The flattened MCMC posterior samples (e.g., from `run_sampler`).
    fit_config : dict
        The fit configuration dictionary, containing parameter labels and setters.
    labels : list of str
        A list of parameter names, in the same order as in `samples`.
    params_template : SolarEclipseParams
        A template `SolarEclipseParams` object. Its `obs_datetime`, `start_datetime`,
        and `end_datetime` attributes will be used for creating model light curves.

    Returns
    -------
    tuple
        - lower_interval : np.ndarray
            The lower 1-sigma uncertainty bound for the model light curve.
        - upper_interval : np.ndarray
            The upper 1-sigma uncertainty bound for the model light curve.
    """
    # Randomly select a subset of samples
    random_indices = np.random.choice(samples.shape[0], size=n_samples, replace=False)
    sampled_params_values = samples[random_indices] # Parameter values for selected samples

    model_fluxes = []
    # Create a new SolarEclipseModel for each sampled parameter set
    # to avoid modifying the original 'params_template' in place if it's used elsewhere.
    # We copy the template and then apply the sampled parameters.
    # Note: `params_template` must have its time array configured correctly for model creation.
    # For light curve generation, a dummy num_entries=1 is fine if obs_datetime is present.
    # Or, if start_datetime/end_datetime are used, then num_entries should be appropriate.
    # Assuming params_template already has `obs_datetime` or `start_datetime`/`end_datetime` set up.
    model_template = SolarEclipseModel(params_template, 1) # Use 1 as num_entries for fixed obs_datetime

    for theta in sampled_params_values:
        # Create a deep copy of params_template to modify for each sample
        # This prevents unintended side effects if `params_template` is used for other purposes
        current_params = SolarEclipseParams()
        for attr in params_template.__dict__:
            if attr not in ['rp']: # Avoid copying property, it will be set by moon_radius
                setattr(current_params, attr, getattr(params_template, attr))

        # Apply the sampled parameter values to the current_params object
        for i, label in enumerate(labels):
            if label in fit_config: # Ensure setter exists for the label
                fit_config[label]["setter"](current_params, theta[i])

        # Generate light curve for the current parameter set
        flux = model_template.light_curve(current_params)
        model_fluxes.append(flux)

    model_fluxes = np.array(model_fluxes)
    # Calculate 16th and 84th percentiles along the time axis
    lower_interval = np.percentile(model_fluxes, 16, axis=0)
    upper_interval = np.percentile(model_fluxes, 84, axis=0)

    return lower_interval, upper_interval


def plot_custom_corner(samples, labels, results, fit_config, model_name, date_str, color_channel, output_dir):
    """
    Generate a customized corner plot for MCMC posteriors.

    This function uses the `corner` library to create a plot showing the
    1D and 2D marginalized posterior distributions of the fitted parameters.
    It customizes titles to include the median and 1-sigma credible intervals.

    Parameters
    ----------
    samples : np.ndarray
        Flattened MCMC samples.
    labels : list of str
        Names of parameters in the order they appear in `samples`.
    results : dict
        Summary statistics for each parameter, including median and uncertainties,
        as returned by `get_fit_summary`.
    fit_config : dict
        Parameter configuration with plot labels and prior information.
    model_name : str
        The name of the model being plotted (e.g., 'linear', 'quadratic').
        Used for the output filename.
    date_str : str
        A date string (e.g., 'YYYYMMDD') for the output filename.
    color_channel : str
        The color channel string (e.g., 'G'). Used to replace a placeholder
        in `plot_label`.
    output_dir : str
        The directory where the corner plot will be saved.
    """
    # Define per-parameter formatting for titles
    custom_formats = {
        "u1": ".2f",
        "u2": ".2f", # Assuming u2 might be present for quadratic
        "atm_ext": ".3f",
        "moon_radius": ".3f",
        "log_sigma_jitter": ".1f"
    }

    # Create a list of format strings for each parameter's title
    title_fmt_list = [custom_formats.get(label, ".5f") for label in labels]

    # True values for the parameters (using median from results)
    truths = [results[label]["median"] for label in labels]

    # Generate the corner plot
    fig = corner.corner(
        samples,
        labels=[fit_config[label]["plot_label"].replace("{color_channel}", color_channel) for label in labels],
        truths=truths,
        quantiles=[0.16, 0.5, 0.84], # Display 16th, 50th (median), and 84th percentiles
        show_titles=True,
        title_fmt=".5f", # Default format for titles (overridden below)
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 12}
    )

    # Manually override each diagonal title with custom formatting and uncertainty
    ndim = len(labels)
    axes = np.array(fig.axes).reshape((ndim, ndim)) # Reshape axes array for easy access
    for i in range(ndim):
        label = labels[i]
        fmt = title_fmt_list[i] # Get custom format for the current parameter
        median = results[label]["median"]
        plus = results[label]["plus_1sigma"]
        minus = results[label]["minus_1sigma"]
        # Retrieve the plot label, replacing the color channel placeholder
        param_label = fit_config[label].get("plot_label", label).replace("{color_channel}", color_channel)
        
        # Set the title for the diagonal subplot (1D marginalized posterior)
        axes[i, i].set_title(
            rf"{param_label} = {median:{fmt}}$^{{+{plus:{fmt}}}}_{{-{minus:{fmt}}}}$",
            fontsize=12
        )

    # Construct the output filename
    filename = os.path.join(output_dir, f"corner_plot_{model_name}_{color_channel}_{date_str}.png")
    fig.savefig(filename, bbox_inches='tight') # Save the figure, ensuring tight layout
    plt.close(fig) # Close the figure to free up memory
    print(f"Corner plot saved to: {filename}")


def plot_data_and_model(results_dict, obs_datetime, obs_flux, color_channel, plot_config, date_str, output_dir):
    """
    Plot observed data, fitted models, and residuals for multiple models.

    Generates a two-panel plot: the top panel shows the normalized observed
    flux and the fitted model light curves, while the bottom panel shows
    the residuals (observed - model). It also includes an optional inset
    plot for a magnified view of a specific time window and shaded regions
    indicating eclipse contact times.

    Parameters
    ----------
    results_dict : dict
        A dictionary where keys are model names and values are dictionaries
        containing `samples`, `labels`, `params`, `fit_config`, and
        `model_plot_config` for each fitted model.
    obs_datetime : np.ndarray
        Array of observation timestamps (datetime objects).
    obs_flux : np.ndarray
        Array of normalized observed flux values.
    color_channel : str
        The color channel string (e.g., 'G'). Used for plot labels.
    plot_config : dict
        General plotting configuration from the `common` section of `config.yaml`,
        including main plot colors, marker styles, and inset settings.
    date_str : str
        A date string (e.g., 'YYYYMMDD') for the output filename.
    output_dir : str
        The directory where the combined plot will be saved.
    """

    plot_color = plot_config.get("plot_color", "k") # Color for observed data points
    plot_marker_symbol = plot_config.get("plot_marker_symbol", ".") # Marker for observed data points
    inset_config = plot_config.get("inset", {}) # Configuration for the inset plot

    # Retrieve eclipse contact times for plotting shaded regions
    tc1 = plot_config.get("eclipse_times", {}).get("tc1")
    tc2 = plot_config.get("eclipse_times", {}).get("tc2")
    tc3 = plot_config.get("eclipse_times", {}).get("tc3")
    tc4 = plot_config.get("eclipse_times", {}).get("tc4")

    # Create a figure with two subplots, sharing the x-axis, with defined height ratios
    fig, (ax, ax_resid) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    # Plot observed flux in the top panel
    ax.plot(obs_datetime, obs_flux, plot_marker_symbol, color=plot_color, label=f'Norm. {color_channel} flux', zorder=5)


    model_fluxes_cache = {} # Cache to store model fluxes for efficient use in inset
    model_times_cache = {}  # Cache to store model times for efficient use in inset

    for model_name, result in results_dict.items():
        # Instantiate SolarEclipseModel with the fitted parameters
        # `num_entries=500` is used here to generate a smooth model curve
        model = SolarEclipseModel(result['params'], 500)
        model_flux = model.light_curve(result['params'])
        model_time = Time(result['params'].time_array, format='jd').to_datetime()

        model_fluxes_cache[model_name] = model_flux
        model_times_cache[model_name] = model_time

        model_plot_config = result.get('model_plot_config', {})
        line_style = model_plot_config.get("line_style", "-") # Line style for the model
        color = model_plot_config.get("color", "k") # Color for the model line
        label = model_plot_config.get("model_label", "model") # Label for the model in legend
        residuals_marker = model_plot_config.get("residuals_marker", ".") # Marker for residuals

        # Plot model light curve in the top panel
        ax.plot(model_time, model_flux, line_style, color=color, label=label, linewidth=1.5, zorder=10)

        # Interpolate model flux to observation times for residual calculation
        model_interp = np.interp(mdates.date2num(obs_datetime), mdates.date2num(model_time), model_flux)
        residuals = obs_flux - model_interp
        # Plot residuals in the bottom panel
        ax_resid.plot(obs_datetime, residuals, residuals_marker, color=color, alpha=0.7)

    ax.set_ylabel("Normalized Flux")
    ax.legend(loc='lower left', frameon=True, fontsize=10) # Legend for the top panel
    ax_resid.axhline(0, color='k', linestyle='--', linewidth=1) # Zero line for residuals
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) # Format x-axis ticks
    ax_resid.set_xlabel("Time [UT]")
    ax_resid.set_ylabel("Residuals")
    ax_resid.ticklabel_format(axis='y', style='sci', scilimits=(-2,2)) # Scientific notation for residuals

    # Plot shaded regions for eclipse contact times in main plot
    if tc1 and tc4:
        tc1_datetime = pd.to_datetime(tc1)
        tc4_datetime = pd.to_datetime(tc4)
        ax.axvspan(tc1_datetime, tc4_datetime, alpha=0.2, color='lightsteelblue', label='Partial Eclipse', zorder=0)
        ax_resid.axvspan(tc1_datetime, tc4_datetime, alpha=0.2, color='lightsteelblue', zorder=0)

    if tc2 and tc3:
        tc2_datetime = pd.to_datetime(tc2)
        tc3_datetime = pd.to_datetime(tc3)
        ax.axvspan(tc2_datetime, tc3_datetime, alpha=0.3, color='mediumpurple', label='Total Eclipse', zorder=0)
        ax_resid.axvspan(tc2_datetime, tc3_datetime, alpha=0.3, color='mediumpurple', zorder=0)

    # Inset plot functionality
    inset_enabled = inset_config.get("enabled", False)
    if inset_enabled:
        # Map string transformation keys to matplotlib transform objects
        transform_map = {
            "axes": ax.transAxes,
            "figure": fig.transFigure, # Use fig.transFigure for figure transform
            "data": ax.transData
        }

        # Retrieve inset plot configuration
        width = inset_config.get("width", "80%")
        height = inset_config.get("height", "80%")
        loc = inset_config.get("loc", "lower left")
        bbox_to_anchor = inset_config.get("bbox_to_anchor", [0.075, 0.30, 0.30, 0.30])
        transform_str = inset_config.get("bbox_transform", "axes")
        bbox_transform = transform_map.get(transform_str, ax.transAxes)

        # Create the inset axes
        ax_ins = inset_axes(ax, width=width, height=height, loc=loc,
                            bbox_to_anchor=bbox_to_anchor, bbox_transform=bbox_transform,
                            borderpad=1) # Add borderpad for spacing

        # Plot observed data in the inset
        ax_ins.plot(obs_datetime, obs_flux, plot_marker_symbol, color=plot_color, markersize=5, alpha=0.7)

        # Plot shaded regions in the inset
        if tc1 and tc4:
            ax_ins.axvspan(tc1_datetime, tc4_datetime, alpha=0.2, color='lightsteelblue')
        if tc2 and tc3:
            ax_ins.axvspan(tc2_datetime, tc3_datetime, alpha=0.3, color='mediumpurple')

        # Plot model light curves in the inset
        for model_name, result in results_dict.items():
            model_flux = model_fluxes_cache[model_name]
            model_time = model_times_cache[model_name]

            model_plot_config = result['model_plot_config']
            line_style = model_plot_config.get("line_style", "-")
            color = model_plot_config.get("color", "k")

            ax_ins.plot(model_time, model_flux, line_style, color=color, linewidth=1)

        # Set x and y limits for the inset plot
        x1_inset = pd.to_datetime(plot_config.get("inset").get("x_lim")[0])
        x2_inset = pd.to_datetime(plot_config.get("inset").get("x_lim")[1])
        y1_inset = plot_config.get("inset").get("y_lim")[0]
        y2_inset = plot_config.get("inset").get("y_lim")[1]
        ax_ins.set_xlim(x1_inset, x2_inset)
        ax_ins.set_ylim(y1_inset, y2_inset)

        # Customize inset ticks and labels
        ax_ins.tick_params(axis='both', labelsize=8)
        ax_ins.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_ins.xaxis.set_major_locator(mdates.MinuteLocator(interval=2)) # Set major ticks every 2 minutes
        ax_ins.set_title('Inset View', fontsize=9) # Add a title to the inset

        # Mark the inset area on the parent axes
        loc1 = plot_config.get("inset").get("mark_inset_loc")[0]
        loc2 = plot_config.get("inset").get("mark_inset_loc")[1]
        mark_inset(ax, ax_ins, loc1=loc1, loc2=loc2, fc="none", ec="0.5", lw=0.8) # Draw connecting lines

    plt.tight_layout() # Adjust layout to prevent overlapping elements
    # Save the figure
    filename = os.path.join(output_dir, f"model_fit_{color_channel}_{date_str}.png")
    plt.savefig(filename, dpi=300) # Save with high resolution
    plt.close(fig) # Close the figure
    print(f"Combined data-model plot saved to: {filename}")
