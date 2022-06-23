# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 23:01:14 2022

@author: Christian Konstantinov
"""

import numpy as np

def to_float32(input_sig: np.ndarray) -> np.ndarray:
    """Return input_sig as an array of 32 bit floats.

    Parameters
    ----------
    input_sig : np.ndarray
        The original signal.

    Returns
    -------
    np.ndarray
        The array with dtype=np.float32
        
    """
    if input_sig.dtype != np.float32:
        return input_sig.astype(np.float32)
    return input_sig

def peak_normalize(input_sig: np.ndarray) -> None:
    """Normalize input_sig in-place to [-1, 1] using the calculated peaks.

    Parameters
    ----------
    input_sig : np.ndarray
        The original signal.

    Returns
    -------
    None.

    """
    if (max_abs := np.max(np.abs(input_sig))) == 0.0:
        return
    input_sig *= 1.0 / max_abs

def rms_normalize(input_sig: np.ndarray, output_sig: np.ndarray) -> None:
    """Normalize output_sig in-place to the rms value of input_sig.

    Parameters
    ----------
    input_sig : numpy.ndarray
        The original signal.
    output_sig : numpy.ndarray
        The signal to normalize.

    Returns
    -------
    None.

    """
    if (rms := np.sqrt(np.mean(np.square(np.sum(output_sig, axis=1))))) == 0.0:
        return
    output_sig *= np.sqrt(np.mean(np.square(input_sig))) / rms

def mono_to_stereo(input_sig: np.ndarray) -> np.ndarray:
    """Convert a mono signal of shape (n,) to a stereo signal of shape (n, 2).

    Parameters
    ----------
    input_sig : np.ndarray
        The input signal.

    Returns
    -------
    np.ndarray
        The output stereo signal.

    """
    _check_mono(input_sig)
    return np.column_stack((input_sig, input_sig))

def stereo_to_mono(input_sig: np.ndarray) -> np.ndarray:
    """Convert a stereo signal of shape (n, 2) to a mono signal of shape (n,).

    Parameters
    ----------
    input_sig : np.ndarray
        The original stereo signal.

    Returns
    -------
    np.ndarray
        The output mono signal.

    """
    _check_stereo(input_sig)
    return (input_sig[:, 0] + input_sig[:, 1]) * 0.5

def LR_to_MS(input_sig: np.ndarray) -> np.ndarray:
    """Given a Left-Right stereo signal, return its Mid-Side equivalent.

    Converts LR to MS with the formula:
        M = (L + R) / 2
        S = (L - R) / 2
    Requires L as channel 0 and R as channel 1.
    Encodes M into channel 0 and S into channel 1.

    Parameters
    ----------
    input_sig : np.ndarray
        The original LR stereo signal.

    Returns
    -------
    np.ndarray
        The output stereo signal in Mid-Side domain.

    """
    _check_stereo(input_sig)
    M = (input_sig[:, 0] + input_sig[:, 1]) * 0.5
    S = (input_sig[:, 0] - input_sig[:, 1]) * 0.5
    return np.column_stack((M, S))

def MS_to_LR(input_sig: np.ndarray) -> np.ndarray:
    """Given a Mid-Side stereo signal return its Left-Right equivalent.

    Converts MS to LR with the formula:
        L = M + S
        R = M - S
    Requires M as channel 0 and S as channel 1.
    Encodes L as channel 0 and R as channel 1.

    Parameters
    ----------
    input_sig : np.ndarray
        The original MS stereo signal.

    Returns
    -------
    np.ndarray
        The output stereo signal in Left-Right domain.

    """
    _check_stereo(input_sig)
    L = input_sig[:, 0] + input_sig[:, 1]
    R = input_sig[:, 0] - input_sig[:, 1]
    return np.column_stack((L, R))

def _check_mono(input_sig: np.ndarray) -> None:
    """If the input signal is not a mono signal, raise an error."""
    if input_sig.ndim != 1:
        raise ValueError(f'Input shape invalid: Expected shape (num samples,), but got shape {input_sig.shape}.')

def _check_stereo(input_sig: np.ndarray) -> None:
    """If the input signal is not a stereo signal, raise an error."""
    if input_sig.ndim != 2 or input_sig.shape[1] != 2:
        raise ValueError(f'Input shape invalid: Expected shape (num samples, 2), but got shape {input_sig.shape}.')
