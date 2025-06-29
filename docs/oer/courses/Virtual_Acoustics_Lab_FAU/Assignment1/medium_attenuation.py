"""Air attenuation calculation."""
import numpy as np
import pyfar as pf

def air_attenuation(
        temperature, frequencies, relative_humidity,
        atmospheric_pressure=101325, saturation_vapor_pressure=None):
    r"""Calculate the pure tone attenuation of sound in air according to
    ISO 9613-1.

    Calculation is in accordance with ISO 9613-1 [#]_. The cshape of the
    outputs is broadcasted from the shapes of the ``temperature``,
    ``relative_humidity``, and ``atmospheric_pressure``.

    Parameters
    ----------
    temperature : float, array_like
        Temperature in degree Celsius.
        Must be in the range of -20°C to 50°C for accuracy of +/-10% or
        must be greater than -70°C for accuracy of +/-50%.
    frequencies : float, array_like
        Frequency in Hz. Must be greater than 50 Hz.
        Just one dimensional array is allowed.
    relative_humidity : float, array_like
        Relative humidity in the range from 0 to 1.
    atmospheric_pressure : int, optional
        Atmospheric pressure in pascal, by default 101325 Pa.
    saturation_vapor_pressure : float, array_like, optional
        Saturation vapor pressure in Pa.
        If not given, the function
        :py:func:`~pyfar.constants.saturation_vapor_pressure` is used.
        Note that the valid temperature range is therefore also dependent on
        :py:func:`~pyfar.constants.saturation_vapor_pressure`.

    Returns
    -------
    alpha : :py:class:`~pyfar.classes.FrequencyData`
        Pure tone air attenuation coefficient in decibels per meter for
        atmospheric absorption.
    m : :py:class:`~pyfar.classes.FrequencyData`
        Pure tone air attenuation coefficient per meter for
        atmospheric absorption. The parameter ``m`` is calculated as
        :math:`m = 10 \cdot \log(\exp(1)) \cdot \alpha`.
    accuracy : :py:class:`~pyfar.classes.FrequencyData`
        accuracy of the results according to the standard:

        ``10``, +/- 10% accuracy
            - molar concentration of water vapour: 0.05% to 5 %.
            - air temperature: 253.15 K to 323.15 (-20 °C to +50°C)
            - atmospheric pressure: less than 200 000 Pa (2 atm)
            - frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa.

        ``20``, +/- 20% accuracy
            - molar concentration of water vapour: 0.005 % to 0.05 %,
              and greater than 5%
            - air temperature: 253.15 K to 323.15 (-20 °C to +50°C)
            - atmospheric pressure: less than 200 000 Pa (2 atm)
            - frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa.

        ``50``, +/- 50% accuracy
            - molar concentration of water vapour: less than 0.005%
            - air temperature: greater than 200 K (- 73 °C)
            - atmospheric pressure: less than 200 000 Pa (2 atm)
            - frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa.

        ``-1``, no valid result
            else.

    References
    ----------
    .. [#] ISO 9613-1:1993, Acoustics -- Attenuation of sound during
           propagation outdoors -- Part 1: Calculation of the absorption of
           sound by the atmosphere.
    """
    # check inputs
    if not isinstance(temperature, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'temperature must be a number or array of numbers')
    if not isinstance(frequencies, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'frequencies must be a number or array of numbers')
    if not isinstance(
            relative_humidity, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'relative_humidity must be a number or array of numbers')
    if np.array(frequencies).ndim > 1:
        raise ValueError('frequencies must be one dimensional.')
    if not isinstance(
            atmospheric_pressure, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'atmospheric_pressure must be a number or array of numbers')

    # check if broadcastable
    try:
        _ = np.broadcast_shapes(
            np.atleast_1d(temperature).shape,
            np.atleast_1d(relative_humidity).shape,
            np.atleast_1d(atmospheric_pressure).shape)
    except ValueError as e:
        raise ValueError(
            'temperature, relative_humidity, and atmospheric_pressure must '
            'have the same shape or be broadcastable.') from e

    # check limits
    if np.any(np.array(temperature) < -73):
        raise ValueError("Temperature must be greater than -73°C.")
    if np.any(np.array(atmospheric_pressure) > 200000):
        raise ValueError("Atmospheric pressure must less than 200 kPa.")

    # convert arrays
    temperature = np.array(
        temperature, dtype=float)[..., np.newaxis]
    relative_humidity = np.array(
        relative_humidity, dtype=float)[..., np.newaxis]
    atmospheric_pressure = np.array(
        atmospheric_pressure, dtype=float)[..., np.newaxis]
    frequencies = np.array(frequencies, dtype=float)

    # calculate air attenuation
    p_atmospheric_ref = 101325
    t_degree_ref = 20

    p_a = atmospheric_pressure
    p_r = p_atmospheric_ref
    f = frequencies
    T = temperature + 273.15
    T_0 = t_degree_ref + 273.15
    T01 = 273.16

    # saturation_vapor_pressure in hPa
    if saturation_vapor_pressure is None:
        saturation_vapor_pressure = calculate_saturation_vapor_pressure(
            temperature)
    p_vapor = relative_humidity*saturation_vapor_pressure

    #    h = p_vapor/p_a*100
    # Iso formula for saturation pressure ratio
    psat_ps0_iso=10**( -6.8346*(T01/T)**1.261+4.6151)

    ps_ps0=p_a/p_r
    # molar concentration of water vapor as a percentage
    h_iso = relative_humidity*psat_ps0_iso/ps_ps0; 

    # Oxygen relaxation frequency (Eq. 3)
    f_rO = (p_a/p_r)*(24+4.04e4*h_iso*(0.02+h_iso)/(0.391+h_iso))

    # Nitrogen relaxation frequency (Eq. 4)
    f_rN = (p_a/p_r)*(T/T_0)**(-1/2)*(9+280*h_iso*np.exp(
        -4.17*((T/T_0)**(-1/3)-1)))

    # air attenuation (Eq. 5)
    air_attenuation = -8.686*f**2*((1.84e-11*p_r/p_a*(T/T_0)**(1/2)) + \
        (T/T_0)**(-5/2)*(0.01275*np.exp(-2239.1/T)*(f_rO + (f**2/f_rO))**(-1)
        +0.1068*np.exp(-3352/T) * (f_rN + (f**2/f_rN))**(-1)))

    alpha = pf.FrequencyData(
        air_attenuation, frequencies=frequencies)

    return alpha


def _calculate_accuracy(
        concentration_water_vapour, temperature, atmospheric_pressure,
        frequencies, shape):
    """Calculate the accuracy of the air attenuation calculation.

    Parameters
    ----------
    concentration_water_vapour : float, array_like
        Molar concentration of water vapor as a percentage.
        Must be between 0% and 100%.
    temperature : float, array_like
        Temperature in degree Celsius.
        Must be above -273.15°C.
    atmospheric_pressure : float, array_like
        Atmospheric pressure in pascal.
        Must be above 0Pa.
    frequencies : float, array_like
        Frequency in Hz.
        Must be lager than 0 Hz.
    shape : tuple
        Shape of the output.

    Returns
    -------
    accuracy : :py:class:`~pyfar.classes.FrequencyData`
        accuracy of the results according to the standard:

            ``10``, +/- 10% accuracy
                - molar concentration of water vapour: 0.05% to 5 %.
                - air temperature: 253.15 K to 323.15 (-20 °C to +50°C)
                - atmospheric pressure: less than 200 000 Pa (2 atm)
                - frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa.

            ``20``, +/- 20% accuracy
                - molar concentration of water vapour: 0.005 % to 0.05 %,
                  and greater than 5%
                - air temperature: 253.15 K to 323.15 (-20 °C to +50°C)
                - atmospheric pressure: less than 200 000 Pa (2 atm)
                - frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa.

            ``50``, +/- 50% accuracy
                - molar concentration of water vapour: less than 0.005%
                - air temperature: greater than 200 K (- 73 °C)
                - atmospheric pressure: less than 200 000 Pa (2 atm)
                - frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa.

            ``-1``, no valid result
                else.
    """
    if np.any(np.array(concentration_water_vapour) < 0) or np.any(
            np.array(concentration_water_vapour) > 100):
        raise ValueError(
            r"Concentration of water vapour must be between 0% and 100%.")
    if np.any(np.array(temperature) < -273.15):
        raise ValueError(
            "Temperature must be greater than -273.15°C.")
    if np.any(np.array(atmospheric_pressure) < 0):
        raise ValueError(
            "Atmospheric pressure must be greater than 0 Pa.")
    if np.any(np.array(frequencies) < 0):
        raise ValueError(
            "Frequencies must be positive.")

    # broadcast inputs
    atmospheric_pressure = np.broadcast_to(atmospheric_pressure, shape)
    h_water_vapor = np.broadcast_to(concentration_water_vapour, shape)
    frequency_pressure_ratio = frequencies/atmospheric_pressure
    accuracy = np.zeros(shape) - 1

    # atmospheric pressure < 200 kPa
    atm_mask = atmospheric_pressure < 200000
    # frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa
    frequency_pressure_ratio_mask = (4e-4 <= frequency_pressure_ratio) & (
        frequency_pressure_ratio <= 10)
    common_mask = atm_mask & frequency_pressure_ratio_mask

    # molar concentration of water vapour: 0.05% to 5 %
    vapor_10_mask = (0.05 <= h_water_vapor) & (h_water_vapor <= 5)
    # molar concentration of water vapour: 0.005% to 0.05 % and greater than 5%
    vapor_20_mask = (5 < h_water_vapor) | (
        (0.005 <= h_water_vapor) & (h_water_vapor < 0.05))
    # molar concentration of water vapour: less than 0.005%
    vapor_50_mask = (0.005 > h_water_vapor)

    # air temperature: 253,15 K to 323,15 (-20 °C to +50°C)
    temp_20_mask = (-20 <= temperature) & (temperature <= 50)
    # air temperature: greater than 200 K (- 73 °C)
    temp_50_mask = (-73 <= temperature)

    # apply masks
    accuracy_50 = common_mask & temp_50_mask & (
        vapor_10_mask | vapor_20_mask | vapor_50_mask)
    accuracy[accuracy_50] = 50

    accuracy_20 = common_mask & temp_20_mask & (
        vapor_10_mask | vapor_20_mask)
    accuracy[accuracy_20] = 20

    accuracy_10 = vapor_10_mask & common_mask & temp_20_mask
    accuracy[accuracy_10] = 10

    # return FrequencyData object
    return pf.FrequencyData(accuracy, frequencies=frequencies)

"""File containing all speed of sound calculation functions."""
import numpy as np


def speed_of_sound_simple(temperature):
    r"""
    Calculate the speed of sound in air using a simplified version
    of the ideal gas law based on the temperature.

    The calculation follows ISO 9613-1 [#]_ (Formula A.5).

    .. math::

        c = 343.2 \cdot \sqrt{\frac{t + 293.15}{t_0 + 293.15}} \mathrm{m/s}

    where:
        - :math:`t` is the air temperature (°C)
        - :math:`t_0=20\mathrm{°C}` is the reference air temperature (°C)

    Parameters
    ----------
    temperature : float, array_like
        Temperature in degree Celsius from -20°C to +50°C.

    Returns
    -------
    speed_of_sound : float, array_like
        Speed of sound in air in (m/s).

    References
    ----------
    .. [#] ISO 9613-1:1993, Acoustics -- Attenuation of sound during
           propagation outdoors -- Part 1: Calculation of the absorption of
           sound by the atmosphere.
    """
    # input validation
    if np.any(np.array(temperature) < -20) or np.any(
            np.array(temperature) > 50):
        raise ValueError("Temperature must be between -20°C and +50°C.")
    # convert to numpy array if necessary
    temperature = np.array(temperature, dtype=float) if isinstance(
        temperature, list) else temperature

    t_0 = 20
    return 343.2*np.sqrt((temperature+273.15)/(t_0+273.15))

def calculate_saturation_vapor_pressure(temperature):
    r"""
    Calculate the saturation vapor pressure of water in Pa using the
    Magnus formula.

    The Magnus formula is valid for temperatures between -45°C and 60°C [#]_.

    .. math::

        e_s = 610.94 \cdot \exp\left(\frac{17.625 \cdot T}{T + 243.04}\right)


    Parameters
    ----------
    temperature : float, array_like
        Temperature in degrees Celsius (°C).

    Returns
    -------
    p_sat : float, array_like
        Saturation vapor pressure in Pa.

    References
    ----------
    .. [#] O. A. Alduchov and R. E. Eskridge, "Improved Magnus Form
           Approximation of Saturation Vapor Pressure," Journal of Applied
           Meteorology and Climatology, vol. 35, no. 4, pp. 60-609, Apr. 1996
    """

    if not isinstance(temperature, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'temperature must be a number or array of numbers')
    if np.any(np.array(
            temperature) < -45) or np.any(np.array(temperature) > 60):
        raise ValueError("Temperature must be in the range of -45°C and 60°C.")
    if isinstance(temperature, (np.ndarray, list, tuple)):
        temperature = np.asarray(temperature, dtype=float)

    return 100 * 6.1094 * np.exp(
        (17.625 * temperature) / (temperature + 243.04))

