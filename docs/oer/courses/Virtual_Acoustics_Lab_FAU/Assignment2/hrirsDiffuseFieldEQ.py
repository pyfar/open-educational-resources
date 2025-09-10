import numpy as np
from scipy.signal import firwin2, hilbert, minimum_phase
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

def rceps(x):
    """
    Compute the real cepstrum (xhat) and minimum-phase reconstruction (yhat) of a real signal x.
    
    Parameters:
        x (array_like): Input real-valued 1D signal.

    Returns:
        xhat (ndarray): Real cepstrum of the input.
        yhat (ndarray): Minimum-phase sequence having the same real cepstrum.
    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("Input must be a 1D real-valued signal.")

    N = len(x)
    is_row = x.ndim == 1

    # Real cepstrum computation
    spectrum = fft(x)
    mag = np.abs(spectrum)
    
    if np.any(mag == 0):
        raise ValueError("FFT magnitude has zeros, can't take log.")

    log_mag = np.log(mag)
    xhat = np.real(ifft(log_mag))

    # Minimum-phase reconstruction
    odd = N % 2
    wn = np.zeros(N)
    wn[0] = 1
    wn[1:(N+odd)//2] = 2
    if odd == 0:
        wn[N//2] = 1  # Nyquist component for even-length signal

    cepstrum_mod = wn * xhat
    yhat = np.real(ifft(np.exp(fft(cepstrum_mod))))

    return xhat, yhat

def hrirsDiffuseFieldEQ(hrirs, min_phase=True, grid_weights=None):
    """
    Calculate diffuse field (common transfer function) EQ.
    
    Parameters:
        hrirs (ndarray): HRIRs of shape [len, 2, grid]
        min_phase (bool, optional): Whether to enforce minimum phase, default True
        grid_weights (ndarray, optional): Weights of shape [grid, 1], default assumes regular grid
    
    Returns:
        eq_taps (ndarray): Filter taps of length len
    """
    num_grid, _, num_taps  = hrirs.shape
    eps = 1e-30
    
    # Default grid weights
    if grid_weights is None:
        grid_weights = np.full((num_grid, 1), (4 * np.pi) / num_grid)
    else:
        assert grid_weights.shape == (num_grid, )
    
    # FFT transform
    nfft = 16 * num_taps  # Interpolation
    H = np.fft.fft(hrirs, n=nfft, axis=2)
    Hs = H[:, :, :nfft // 2 + 1]
    
    # Weighted RMS
    Havg = np.sqrt(np.sum(grid_weights[:, np.newaxis, np.newaxis] * (np.abs(Hs) ** 2), axis=0)/ (4 * np.pi))
    Havg = np.mean(Havg, axis=0)  # Average over left and right
    
    # Smoothing
    Havg_smooth = Havg.copy()
    for bin in range(1, nfft // 2 + 1):
        if bin % 2 == 1:
            avg_idx = np.arange(max(bin - bin // 2, 0), min(int(3 / 2 * bin), nfft // 2 + 1))
        win = np.hanning(len(avg_idx)) + eps
        Havg_smooth[bin] = np.sum(win * Havg[avg_idx]) / np.sum(win)
    
    # Frequency mask
    freq_weight = np.ones(nfft // 2 + 1)
    w_lo = np.hanning(nfft // 64 + 1)
    freq_weight[:nfft // 128 + 1] = w_lo[:nfft // 128 + 1]
    w_hi = np.hanning(nfft // 2 + 1)
    freq_weight[-(nfft // 4):] = w_hi[-(nfft // 4):]
    
    # Frequency weighted inversion
    Hinv_weighted = freq_weight * (1 / Havg_smooth) + (1 - freq_weight)
    
    # FIR filter design
    eq_taps = firwin2(num_taps + 2, np.linspace(0, 1, nfft // 2 + 1), Hinv_weighted,window='hamming')
    
    # Minimum-phase transformation
    if min_phase:
        _, eq_taps = rceps(eq_taps)
    
    return eq_taps
