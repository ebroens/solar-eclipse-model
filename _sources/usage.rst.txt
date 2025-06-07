.. _usage:

Usage Instructions
==================

This section provides guidance on how to use the `Solar Eclipse Model` API components, and then on the full MCMC analysis workflow.

Installation
------------

First, ensure you have the necessary dependencies installed. It is recommended to use a Python virtual environment.

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/your-username/solar-eclipse-model.git
    cd solar-eclipse-model

    # Create and activate a virtual environment
    python -m venv venv_name
    source venv_name/bin/activate # On Windows: .\venv_name\Scripts\activate

    # Install run-time dependencies
    pip install numpy pandas matplotlib astropy emcee corner pyyaml batman-package dill

    # Install for documentation generation (optionally)
    pip install sphinx sphinx-rtd-theme furo


Basic API Usage
---------------

The core functionality is exposed through the Python API, allowing you to
programmatically interact with the solar eclipse model. The primary classes are
:py:class:`custom_models.solareclipsemodel.SolarEclipseModel` and
:py:class:`custom_models.solareclipsemodel.SolarEclipseParams`.

Here's a basic example of how to instantiate the model and generate a light curve:

.. plot::
   :include-source:

   from custom_models.solareclipsemodel import SolarEclipseModel, SolarEclipseParams
   from astropy.time import Time
   import astropy.units as u
   import numpy as np
   import matplotlib.pyplot as plt
   import matplotlib.dates as mdates
   import pandas as pd

   # 1. Define observation parameters
   params = SolarEclipseParams()
   params.limb_dark = "linear" # Choose a limb darkening law
   params.u = [0.3] # Limb darkening coefficient(s)
   params.atm_ext = 0.08 # Atmospheric extinction coefficient
   params.lat = -29.88606 # Observer latitude (e.g., for the 2019 total eclipse in Chile)
   params.lon = -70.68380 # Observer longitude

   # 2. Define the time array for the light curve
   # Using a simple linear spacing for demonstration
   start_time_str = "2019-07-02 19:00:00"
   end_time_str = "2019-07-02 22:00:00"
   num_time_points = 500
   params.start_datetime = start_time_str
   params.end_datetime = end_time_str

   # 3. Instantiate the SolarEclipseModel
   # Note: If params.obs_datetime is not None, num_entries is ignored.
   # Otherwise, it's used to generate time_array between start_datetime and end_datetime.
   model = SolarEclipseModel(params, num_time_points)

   # 4. Generate the light curve
   synthetic_flux = model.light_curve(params)

   # 5. Plot the light curve
   # Convert Julian Date time_array back to datetime objects for plotting
   plot_times = Time(model.time_array, format='jd').to_datetime()

   plt.figure(figsize=(10, 5))
   plt.plot(plot_times, synthetic_flux, color='blue', linewidth=2)
   ax = plt.gca()
   ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
   plt.xlabel("Time (UT)")
   plt.ylabel("Normalized Flux")
   plt.title(f"Synthetic Light Curve (Limb Darkening: {params.limb_dark.capitalize()})")
   plt.grid(True)
   plt.tight_layout()
   plt.show()


Full MCMC Analysis Workflow
---------------------------

For details on running the full MCMC fitting pipeline using the
`model_eclipse_mcmc.py` script, including command-line options and analysis
of typical outputs (like corner plots and data-model comparisons), please
refer to the :doc:`examples` section.
