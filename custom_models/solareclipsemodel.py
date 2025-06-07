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
solareclipsemodel.py

Model solar eclipse 'light curves' using the BATMAN transit model and astropy ephemerides.
This module defines:
- SolarEclipseModel: a subclass of TransitModel from `batman` adapted for solar/lunar geometry.
- SolarEclipseParams: an extension of `TransitParams` with observer location and atmospheric extinction.
"""

from batman.transitmodel import TransitModel, TransitParams
from astropy.coordinates import solar_system_ephemeris, EarthLocation, get_body, Angle, AltAz
from astropy.time import Time
import astropy.units as u
from astropy.constants import R_sun, R_earth
import numpy as np


class SolarEclipseModel(TransitModel):
    """
    Transit model adapted to simulate solar eclipses as seen from Earth.

    This class extends the `batman.TransitModel` to incorporate the specific
    geometry and ephemerides required for modeling solar eclipses from an
    Earth-based observer. It calculates the angular separation between the
    Moon and Sun, their apparent sizes, and accounts for atmospheric extinction.

    Attributes
    ----------
    time_array : np.ndarray
        Observation times in Julian Date (JD). This array is generated either
        from `start_datetime` and `end_datetime` or directly from `obs_datetime`.
    sun : astropy.coordinates.SkyCoord
        Ephemeris-based celestial position of the Sun over the `time_array`,
        as observed from the specified Earth location.
    moon : astropy.coordinates.SkyCoord
        Ephemeris-based celestial position of the Moon over the `time_array`,
        as observed from the specified Earth location.
    ds : astropy.units.Quantity
        Angular separation between the Moon and Sun, normalized by the Sun's
        apparent angular radius. This dimensionless ratio is crucial for the
        `batman` transit model calculation.
    """

    def __init__(self, params, num_entries: int = None):
        """
        Initialize SolarEclipseModel with ephemerides and geometry.

        The time array for the model can be provided either as a fixed array
        of observation datetimes (`params.obs_datetime`) or generated
        linearly between a start and end datetime (`params.start_datetime`,
        `params.end_datetime`) with a specified number of entries.

        Parameters
        ----------
        params : SolarEclipseParams
            An instance of `SolarEclipseParams` containing the necessary
            input parameters, including observer location, limb darkening
            coefficients, atmospheric extinction, and time information.
            If the Moon radius is not provided it will be calculated.
        num_entries : int, optional
            The number of time samples to generate if `params.start_datetime`
            and `params.end_datetime` are provided. If `params.obs_datetime`
            is used, this parameter is effectively ignored.
        """
        if params.start_datetime and params.end_datetime and num_entries:
            # Generate a time array from start to end datetime
            self.time_array = self._generate_time_array(params.start_datetime, params.end_datetime, num_entries)
            params.time_array = self.time_array # Update params object with the generated time array
        elif params.obs_datetime is not None:
            # Use provided observation datetimes, converting to Julian Date
            self.time_array = Time(params.obs_datetime).jd
            params.time_array = self.time_array # Ensure params object has the time array
        else:
            raise ValueError("Must provide either (start_datetime, end_datetime, num_entries) or obs_datetime.")

        if params.moon_radius is None:
            # Get Sun and Moon ephemerides from observer location
            self.sun = self._get_sun_data(params)
            self.moon = self._get_moon_data(params)
            params.moon_radius = self.relative_moon_size(params)

        # Initialize the base TransitModel class
        super().__init__(params, self.time_array)

        # Set the ephemeris kernel for astropy (de432s is recommended for planetary positions)
        solar_system_ephemeris.set('de432s')
        
        # Get Sun and Moon ephemerides from observer location
        self.sun = self._get_sun_data(params)
        self.moon = self._get_moon_data(params)

        # Compute the normalized angular separation between Moon and Sun
        self.ds = self._compute_moon_sun_dst(params)


    def _generate_time_array(self, start_time: str, end_time: str, num_entries: int) -> np.ndarray:
        """
        Generates a linearly spaced array of Julian Dates between a start and end time.

        Parameters
        ----------
        start_time : str
            The start time for the array, in a format parseable by `astropy.time.Time`.
        end_time : str
            The end time for the array, in a format parseable by `astropy.time.Time`.
        num_entries : int
            The number of time samples to generate between `start_time` and `end_time`.

        Returns
        -------
        np.ndarray
            A NumPy array of Julian Dates.
        """
        return np.linspace(Time(start_time).jd, Time(end_time).jd, num_entries)

    def _get_sun_data(self, params) -> object:
        """
        Retrieves solar ephemerides from the specified observer location.

        Parameters
        ----------
        params : SolarEclipseParams
            The parameter object containing observer's latitude and longitude.

        Returns
        -------
        astropy.coordinates.SkyCoord
            An `astropy.coordinates.SkyCoord` object representing the
            Sun's position over the `time_array` from the observer's location.
        """
        loc = EarthLocation.from_geodetic(params.lon, params.lat)
        return get_body("sun", Time(self.time_array, format="jd"), loc)

    def _get_moon_data(self, params) -> object:
        """
        Retrieves lunar ephemerides from the specified observer location.

        Parameters
        ----------
        params : SolarEclipseParams
            The parameter object containing observer's latitude and longitude.

        Returns
        -------
        astropy.coordinates.SkyCoord
            An `astropy.coordinates.SkyCoord` object representing the
            Moon's position over the `time_array` from the observer's location.
        """
        loc = EarthLocation.from_geodetic(params.lon, params.lat)
        return get_body("moon", Time(self.time_array, format="jd"), loc)

    def _compute_moon_sun_dst(self, params) -> u.Quantity:
        """
        Computes the angular separation between the Moon and Sun, normalized by the
        Sun's apparent angular radius.

        This normalized separation is crucial for the `batman` transit model,
        where `ds` effectively represents the distance between the transiting
        body (Moon) and the host star (Sun) in units of the star's radius.

        Parameters
        ----------
        params : SolarEclipseParams
            The parameter object containing observer's latitude and longitude,
            needed for ephemeris calculations.

        Returns
        -------
        astropy.units.Quantity
            A dimensionless `astropy.units.Quantity` representing the ratio of
            the Moon-Sun separation to the Sun's apparent radius.
        """
        sep = self.moon.separation(self.sun).to(u.radian) # Angular separation in radians
        return sep / self._sun_apparent_radius(params) # Normalized by Sun's apparent radius

    def _sun_apparent_radius(self, params) -> Angle:
        """
        Calculates the Sun's apparent angular radius as seen from Earth.

        Parameters
        ----------
        params : SolarEclipseParams
            The parameter object (currently not directly used but included for consistency
            if future parameter dependencies arise).

        Returns
        -------
        astropy.coordinates.Angle
            The apparent angular radius of the Sun in radians.
        """
        # Distance to the Sun from observer (first element used assuming a scalar distance for the first time step)
        sun_dist = self.sun.distance[0].to(u.m).value
        # Calculate apparent radius using Sun's actual radius and distance
        return Angle(R_sun.value / sun_dist, u.radian)

    def relative_moon_size(self, params) -> float:
        """
        Returns the apparent angular radius ratio of the Moon to the Sun.

        This value (`k` in transit terminology) is set as `params.rp` and
        is used by the `batman` model to determine the relative size of
        the transiting body.

        Parameters
        ----------
        params : SolarEclipseParams
            The parameter object containing the `k` value (Moon-to-Earth radius ratio).

        Returns
        -------
        float
            The dimensionless ratio of the Moon's apparent angular radius to
            the Sun's apparent angular radius.
        """
        moon_dist = self.moon.distance[0].to(u.m).value # Distance to the Moon
        # Calculate Moon's apparent radius using Earth's radius and params.k
        # (params.k is ratio of moon radius to Earth radius for ephemeris)
        moon_app_radius = Angle(R_earth.value * params.k / moon_dist, u.radian)
        return (moon_app_radius / self._sun_apparent_radius(params)).value

    def sun_airmass_young(self, params) -> np.ndarray:
        """
        Computes the atmospheric airmass for the Sun's position using Young's formula.

        Airmass quantifies the amount of atmosphere light passes through, affecting
        atmospheric extinction. Young's formula is an empirical approximation.

        Parameters
        ----------
        params : SolarEclipseParams
            The parameter object containing observer's latitude and longitude.

        Returns
        -------
        np.ndarray
            An array of airmass values corresponding to each time step in `time_array`.
        """
        loc = EarthLocation.from_geodetic(params.lon, params.lat)
        # Transform Sun's celestial coordinates to Altitude-Azimuth frame for the observer
        altaz = self.sun.transform_to(AltAz(obstime=Time(self.time_array, format="jd"), location=loc))
        # Calculate zenith angle (90 degrees - altitude)
        z = (90 * u.deg - altaz.alt).to(u.deg).value
        z_rad = np.radians(z) # Zenith angle in radians

        # Young's formula for airmass
        numerator = (1.002432 * (np.cos(z_rad))**2 + 0.148386 * np.cos(z_rad) + 0.0096467)
        denominator = ((np.cos(z_rad))**3 + 0.149864 * (np.cos(z_rad))**2 + 0.0102963 * np.cos(z_rad) + 0.000303978)
        return numerator / denominator

    def light_curve(self, params) -> np.ndarray:
        """
        Generates the normalized solar eclipse light curve, accounting for
        atmospheric extinction.

        This method first computes the basic transit light curve using the
        `batman` model's `light_curve` method (inherited), then applies
        atmospheric extinction based on the calculated airmass and `params.atm_ext`.
        Finally, the light curve is normalized by its maximum flux value.

        Parameters
        ----------
        params : SolarEclipseParams
            The parameter object containing `atm_ext` (atmospheric extinction
            coefficient) and other transit parameters.

        Returns
        -------
        np.ndarray
            An array of normalized flux values representing the simulated
            solar eclipse light curve.
        """
        # Generate the basic transit light curve from the parent class
        flux = super().light_curve(params)
        # Compute airmass for atmospheric extinction
        airmass = self.sun_airmass_young(params)
        # Apply atmospheric extinction: Flux_extincted = Flux_unextincted * exp(-atm_ext * airmass)
        flux *= np.exp(-1 * params.atm_ext * airmass)
        # Normalize the light curve by its maximum value
        return flux / np.max(flux)


class SolarEclipseParams(TransitParams):
    """
    Extended TransitParams for solar eclipse modeling.

    This class inherits from `batman.TransitParams` and adds specific attributes
    relevant to modeling solar eclipses from an Earth-based observer, such as
    observer location, moon-specific parameters, and atmospheric extinction.
    It also redefines the `rp` property to link to `moon_radius`.

    Attributes
    ----------
    lat : float or None
        Latitude of the observer in degrees (e.g., -29.88606 for Chile).
    lon : float or None
        Longitude of the observer in degrees (e.g., -70.68380 for Chile).
    obs_datetime : list or array or None
        Observation times (datetime objects). If provided, the model will use
        these specific times for light curve generation.
    start_datetime : str or None
        Start time for generating a linearly spaced time array (e.g., "YYYY-MM-DD HH:MM:SS").
        Used in conjunction with `end_datetime` and `num_entries` in `SolarEclipseModel`.
    end_datetime : str or None
        End time for generating a linearly spaced time array.
    moon_radius : float
        The radius of the Moon in units of the Sun's radius. This is the parameter
        that corresponds to `rp` (planet radius in stellar radii) in the `batman` context.
    k : float
        Ratio of the Moon's radius to Earth's radius (IAU standard value: 0.2725076).
        This is used in ephemeris calculations related to the Moon's apparent size.
    atm_ext : float
        Atmospheric extinction coefficient. This value quantifies how much
        light is absorbed and scattered by the atmosphere.
    model_name : str or None
        An optional name for the model (e.g., 'linear', 'quadratic'). Useful for logging.
    u : list of floats
        Limb darkening coefficients. The number of elements in this list depends
        on the `limb_dark` law (e.g., `[u1]` for linear, `[u1, u2]` for quadratic).
    limb_dark : str
        The limb darkening law to apply ('uniform', 'linear', 'quadratic').

    Inherited Attributes from `TransitParams` (set to defaults as they are
    not primarily fitted for solar eclipses, but required by `batman`):
    - t0 (float): Central transit time.
    - per (float): Orbital period in days.
    - a (float): Semi-major axis in stellar radii.
    - inc (float): Inclination of the orbit in degrees.
    - ecc (float): Eccentricity of the orbit.
    - w (float): Argument of periastron in degrees.
    """

    def __init__(self):
        super().__init__()
        # Solar Eclipse specific parameters
        self.lat = None
        self.lon = None
        self.obs_datetime = None
        self.start_datetime = None
        self.end_datetime = None
        self.k = 0.2725076  # IAU standard value for Moon radius / Earth radius
        self.atm_ext = 0.01
        self.model_name = None
        self.u = [0.3] # Default limb darkening coefficient for linear
        self.limb_dark = "linear" # Default limb darkening law
        self.moon_radius = None # Default Moon radius in Sun-radius units

        # Required parameters for TransitModel (set to defaults because not directly
        # used or fitted for solar eclipse models in the same way as exoplanet transits).
        # rp is special and linked to moon_radius via a property.
        self.t0 = 0.0  # Default central transit time
        self.per = 1.0  # Default period (days)
        self.a = 10.0  # Default semi-major axis (stellar radii)
        self.inc = 90.0  # Default inclination (degrees)
        self.ecc = 0.0  # Default eccentricity
        self.w = 90.0  # Default argument of periastron (degrees)
        # self.rp is handled by the property below, initializing moon_radius instead

    @property
    def rp(self):
        """
        Planet radius (in stellar radii), linked to `moon_radius`.

        This property ensures that the `rp` attribute required by `batman.TransitParams`
        is always consistent with the `moon_radius` attribute, as the Moon acts
        as the "planet" in this solar eclipse model.
        """
        return self.moon_radius

    @rp.setter
    def rp(self, value):
        """
        Setter for `rp`, which updates `moon_radius`.

        Parameters
        ----------
        value : float
            The new value for the effective planet radius in stellar radii.
        """
        self.moon_radius = value
