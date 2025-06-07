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
model_eclipse_mcmc.py

Main script to run MCMC fitting for solar eclipse light curve models.
Supports YAML configuration, CLI arguments, and multi-model comparisons.
"""

import argparse
import yaml
import os
import numpy as np
import pandas as pd
from datetime import datetime
from custom_models.solareclipsemodel import SolarEclipseModel, SolarEclipseParams
from utils.model_utils import run_sampler, plot_custom_corner, plot_data_and_model, get_fit_summary, update_params_from_results, save_results_to_csv
from astropy.time import Time

def load_config(config_file):
    """
    Load configuration YAML file.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file (e.g., 'config.yaml').

    Returns
    -------
    dict
        A dictionary containing the loaded configuration.
    """
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def create_params(model_config, obs_datetime):
    """
    Construct a SolarEclipsParams object from a model configuration dictionary.

    This function initializes a SolarEclipsParams object and populates its
    attributes based on the provided model configuration and observed timestamps.

    Parameters
    ----------
    model_config : dict
        A dictionary containing the configuration for a single model, typically
        from the 'models' section of the `config.yaml` file.
    obs_datetime : array-like
        Observation timestamps, expected to be in a format convertible to
        datetime64 (e.g., numpy array of datetimes or pandas Timestamp series).
        These times are used to set the `obs_datetime` attribute of the
        SolarEclipsParams object for fixed-time light curve generation.

    Returns
    -------
    SolarEclipsParams
        An initialized SolarEclipsParams object with attributes set according
        to the model configuration.
    """
    params = SolarEclipseParams()

    # Assign model attributes based on config (default values provided if missing)
    # The 'limb_dark' law (e.g., "uniform", "linear", "quadratic")
    params.limb_dark = model_config.get("limb_dark", "uniform")
    # Limb darkening coefficients (e.g., [u1], [u1, u2])
    params.u = model_config.get("u", [])
    # Radius of the Moon in Sun-radius units
    params.moon_radius = model_config.get("moon_radius", 1.0)
    # Atmospheric extinction coefficient
    params.atm_ext = model_config.get("atm_ext", 0.01)
    # Latitude of the observer in degrees
    params.lat = model_config.get("lat")
    # Longitude of the observer in degrees
    params.lon = model_config.get("lon")
    # Observation timestamps
    params.obs_datetime = obs_datetime

    return params


def make_setter(label):
    """
    Creates a setter function for updating SolarEclipsParams attributes.

    This function dynamically creates a callable that sets a specific parameter
    on a `SolarEclipsParams` object. It handles special cases for limb
    darkening coefficients (`u1`, `u2`, etc.) and 'log\_' prefixed parameters.

    Parameters
    ----------
    label : str
        The name of the parameter for which to create a setter (e.g., 'u1',
        'atm_ext', 'log\_sigma\_jitter').

    Returns
    -------
    callable
        A lambda function ``(params, val) -> None`` that sets the specified
        parameter ``val`` on the ``params`` object. Returns a no-op function
        for 'log\_' prefixed labels as they are handled differently during MCMC.
    """
    if label.startswith("u") and label[1:].isdigit():
        # Handle 'u1', 'u2', etc. by setting the appropriate index in params.u
        index = int(label[1:]) - 1  # Convert "u1" to index 0, "u2" to index 1, etc.
        return lambda params, val: _set_u_at_index(params, index, val)
    elif "log_" in label:
        # 'log_' prefixed parameters are typically derived, so no direct setter
        # is needed for the original parameter in the SolarEclipseParams object
        # (e.g., log_sigma_jitter is converted to sigma_jitter during likelihood).
        return lambda params, val: None
    else:
        # Default case: directly set the attribute on the params object
        return lambda params, val: setattr(params, label, val)

def _set_u_at_index(params, index, val):
    """
    Helper function to safely set a limb darkening coefficient (u) at a given index.

    Ensures the `params.u` list is initialized and has sufficient length
    before assigning a value.

    Parameters
    ----------
    params : SolarEclipsParams
        The parameter object to modify.
    index : int
        The index in the `params.u` list to set.
    val : float
        The value to assign to `params.u[index]`.
    """
    # Ensure 'u' exists and has enough length
    if not hasattr(params, "u") or params.u is None:
        params.u = []
    while len(params.u) <= index:
        params.u.append(None) # Extend list with None until the target index is reachable
    params.u[index] = val

def build_fit_config(model_config):
    """
    Builds the MCMC fitting configuration dictionary from a model's definition.

    This function processes the ``fit_params`` section of a model's configuration
    to create a structured dictionary used by the MCMC sampler. It includes
    priors, a setter function for each parameter, and a plotting label.

    Parameters
    ----------
    model_config : dict
        A dictionary representing a single model's configuration, typically
        from the 'models' section of the ``config.yaml`` file. It must contain
        a 'fit_params' key.

    Returns
    -------
    dict
        A dictionary where keys are parameter labels (e.g., 'u1', 'atm_ext')
        and values are dictionaries containing:

        - 'prior' (tuple): A tuple (lower_bound, upper_bound) for the uniform
          prior. For 'log\_' prefixed parameters, these bounds are already in log space.
        - 'setter' (callable): A function that sets the parameter on a
          `SolarEclipsParams` object.
        - 'plot_label' (str): The LaTeX-formatted label for plotting this parameter.

    """
    fit_config = {}
    for label, entry in model_config["fit_params"].items():
        setter = make_setter(label)
        # Convert prior bounds to log space if the label starts with 'log_'
        prior_bounds = tuple(np.log(entry["prior"]) if "log_" in label else entry["prior"])
        fit_config[label] = {
            "prior": prior_bounds,
            "setter": make_setter(label),
            "plot_label": entry["plot_label"]
        }
    return fit_config


def main():
    """
    Main entry point for the solar eclipse MCMC fitting tool.

    This function handles command-line argument parsing, loads the configuration
    from a YAML file, loads observational data, runs the MCMC sampler for the
    specified models, summarizes the fitting results, and generates output plots.
    """
    parser = argparse.ArgumentParser(description="Solar Eclipse MCMC fitting tool.")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                        help="Path to YAML config file.")
    parser.add_argument("--model", type=str, default="all",
                        help="Name of the model to run from the config file, or 'all' to run all models (default: 'all').")
    args = parser.parse_args()

    config = load_config(args.config)
    plot_config = config["common"]["plot_config"]
    data_file = config["common"]["data_file"]
    output_dir = config["common"].get("output_dir", "./results")
    os.makedirs(output_dir, exist_ok=True)

    # Load observational data
    obs_data = pd.read_csv(data_file, delimiter=' ', comment='#')
    # Combine date and time columns into a single datetime column
    obs_data['date_time'] = pd.to_datetime(obs_data[config["common"]["date_column"]]) + \
                            pd.to_timedelta(obs_data[config["common"]["time_column"]])
    obs_datetime = obs_data['date_time'].to_numpy()
    # Normalize the observed flux by its maximum value
    obs_flux = obs_data[config["common"]["flux_column"]] / np.max(obs_data[config["common"]["flux_column"]])
    sigma_data = config["common"]["sigma_data"] # Estimated accuracy of the measured flux

    color_channel = config["common"]["color_channel"] # Used for filenames and corner plots

    # Determine which models to run
    model_names = config["models"].keys() if args.model == "all" else [args.model]
    results_dict = {} # Dictionary to store results for each model

    for name in model_names:
        print(f"\nRunning model: {name}")
        model_cfg = config["models"][name]
        fit_config = build_fit_config(model_cfg)
        params = create_params(model_cfg, obs_datetime)

        nwalkers = 32 # Number of walkers for the MCMC sampler
        nsteps = 1000 # Number of steps for the MCMC sampler

        # Run the MCMC sampler
        samples, log_probs = run_sampler(obs_flux, sigma_data, fit_config, params, nwalkers, nsteps, output_dir, save_data=True)
        labels = list(fit_config.keys()) # Labels for the fitted parameters
        results = get_fit_summary(samples, log_probs, labels) # Summarize MCMC results

        # Save MCMC fit summary to a CSV file
        filename = os.path.join(output_dir, f"mcmc_fit_results_{name}_G_{datetime.now().strftime('%Y%m%d')}.csv")
        save_results_to_csv(results, filename)

        # Update model parameters with the median values from the MCMC fit
        params = update_params_from_results(params, results, fit_config)
        # Set start and end datetimes for model evaluation (e.g., for plotting)
        params.start_datetime = plot_config["plot_range"]["start_datetime"] 
        params.end_datetime = plot_config["plot_range"]["end_datetime"] 

        # Store results for plotting and further analysis
        results_dict[name] = {
            "samples": samples,
            "labels": labels,
            "params": params,
            "fit_config": fit_config,
            "model_plot_config": config.get('models', {}).get(name, {}).get('plotting', {})
        }

        # Generate and save the corner plot for MCMC posteriors
        plot_custom_corner(
            samples,
            labels,
            results,
            fit_config,
            name,
            datetime.now().strftime("%Y%m%d"),
            config.get("common", {}).get("color_channel", "unk"),
            output_dir
        )

    # Generate final data-model comparison plots for all models
    plot_data_and_model(
        results_dict,
        obs_datetime,
        obs_flux,
        color_channel,
        plot_config,
        date_str=datetime.now().strftime("%Y%m%d"),
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
