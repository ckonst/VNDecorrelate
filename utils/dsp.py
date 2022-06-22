# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 23:01:14 2022

@author: Christian Konstantinov
"""

import numpy as np

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
    output_sig *= np.sqrt(np.mean(np.square(input_sig))) \
        /  np.sqrt(np.mean(np.square(np.sum(output_sig, axis=1))))

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
    input_sig *= 1.0 / np.max(input_sig)

def LR_to_MS(input_sig: np.ndarray) -> np.ndarray:
    pass

def MS_to_LR(input_sig: np.ndarray) -> np.ndarray:
    pass