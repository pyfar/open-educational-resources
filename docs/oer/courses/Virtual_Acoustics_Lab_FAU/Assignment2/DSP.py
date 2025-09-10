import numpy as np
import pyfar as pf

class DSP:
    """
    API for DSP units.
    Provided
    """
    
    def __init__(self):
        self.numberOfInputs = 0
        self.numberOfOutputs = 0
        self.blockSize = 256
    
    def checkConfig(self):
        assert self.numberOfInputs > 0, "Set Input number"
        assert self.numberOfOutputs > 0, "Set Output number"
        output = self.process(np.zeros((self.blockSize, self.numberOfInputs)))
       # assert output.shape[1] == self.numberOfOutputs, "Output size mismatch"

def m2smp(m, speed_of_sound, fs):
    # convert meter to samples
    # PROVIDED %
    return (m / speed_of_sound) * fs

def call112(len_samp, fs):
    # Create time array in seconds
    #time_seconds = (0:len_samp-1) / fs

    # Create two complex tones and noise
    f0 = 330 # fundamental frequency of first tone
    f0_fourth = f0*4/3 # fundamental frequency of fourth

    signalFundamental = sawtooth(f0, len_samp, sampling_rate=fs) # ToDo Add to pyfar
    signalFourth = sawtooth(f0_fourth, len_samp, sampling_rate=fs) 

    # fade between two tones
    noteLen_s = 3 / 4
    noteRepetitions = int(np.ceil(len_samp/fs*2*noteLen_s))

    # ToDo implement tukey windows in pyfar
    window = np.ones(int(fs*noteLen_s))
    winFundamental = np.matlib.repmat(np.concatenate((window, np.zeros(int(fs*noteLen_s)))), 1, noteRepetitions)

    winFourth = np.roll(winFundamental,int(-fs*noteLen_s))

    signalFundamentalWind = pf.multiply((signalFundamental , pf.Signal(winFundamental[0, 0:len_samp], fs)), domain="time")
    signalFourthWind = pf.multiply((signalFourth , pf.Signal(winFourth[0, 0:len_samp], fs)), domain="time")

    signal = signalFundamentalWind + signalFourthWind

    # add (road) noise
    noiseGain = 0.5

    noise = pf.signals.noise(len_samp, "white",sampling_rate=fs) * noiseGain

    signal = pf.add((signal,noise),domain="time")

    return signal

#ToDo add this to pyfar
def sawtooth(frequency, n_samples, amplitude=1, phase=0, sampling_rate=44100,
         full_period=False):
    """Generate a single or multi channel sawtooth signal.

    Parameters
    ----------
    frequency : double, array like
        Frequency of the sine in Hz (0 <= `frequency` <= `sampling_rate`/2).
    n_samples : int
        Length of the signal in samples.
    amplitude : double, array like, optional
        The amplitude. The default is ``1``.
    phase : double, array like, optional
        The phase in radians. The default is ``0``.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is ``44100``.
    full_period : boolean, optional
        Make sure that the returned signal contains an integer number of
        periods resulting in a periodic signal. This is done by adjusting the
        frequency of the sawtooth. The default is ``False``.

    Returns
    -------
    signal : Signal
        The sawtooth signal. The Signal is in the time domain and has the ``rms``
        FFT normalization (see :py:func:`~pyfar.dsp.fft.normalization`).
        The exact frequency, amplitude and phase are written to `comment`.

    Notes
    -----
    The parameters `frequency`, `amplitude`, and `phase` are broadcasted using
    the :doc:`numpy rules<numpy:user/basics.broadcasting>`. For example
    `frequency` could be of shape ``(2, 4)``, `amplitude` of shape ``(2, 1)``,
    and `phase` could be a scalar. In this case all parameters would be
    broadcasted to a shape of ``(2, 4)``.
    """

    # check and match the cshape
    try:
        cshape, (frequency, amplitude, phase) = _match_shape(
            frequency, amplitude, phase)
    except ValueError as error:
        raise ValueError(("The parameters frequency, amplitude, and phase can "
                          "not be broadcasted to the same shape")) from error

    if np.any(frequency < 0) or np.any(frequency > sampling_rate/2):
        raise ValueError(
            f"The frequency must be between 0 and {sampling_rate/2} Hz")

    # generate the sawtooth signal
    n_samples = int(n_samples)
    times = np.arange(n_samples) / sampling_rate
    sawtooth = np.zeros(cshape + (n_samples, ))
    for idx in np.ndindex(cshape):
        if full_period:
            # nearest number of full periods
            num_periods = np.round(
                n_samples / sampling_rate * frequency[idx])
            # corresponding frequency
            frequency[idx] = num_periods * sampling_rate / n_samples

        cycles = frequency[idx] * times
        sawtooth[idx] = amplitude[idx] * (2 * (cycles - np.floor(cycles)) - 1)

    # save to Signal
    nl = "\n"  # required as variable because f-strings cannot contain "\"
    comment = (f"Sine signal (f = {str(frequency).replace(nl, ',')} Hz, "
               f"amplitude = {str(amplitude).replace(nl, ',')}, "
               f"phase = {str(phase).replace(nl, ',')} rad)")

    signal = pf.Signal(
        sawtooth, sampling_rate, fft_norm="rms", comment=comment)
    
    return signal
    


def _match_shape(*args):
    """
    Match the shape of *args to the shape of the arg with the largest size
    using np.broadcast_shapes and np.broadcast_to().

    Parameters
    ----------
    *args :
        data for matching shape

    Returns
    -------
    shape : tuple
        new common shape of the args
    args : list
        args with new common shape
        (*arg_1, *arg_2, ..., *arg_N)
    """

    # broadcast shapes
    shape = np.broadcast_shapes(*[np.atleast_1d(arg).shape for arg in args])

    # match the shape
    result = []
    for arg in args:
        arg = np.broadcast_to(arg, shape)
        arg.setflags(write=1)
        result.append(arg)

    return shape, result
    