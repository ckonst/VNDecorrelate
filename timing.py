# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 18:45:12 2022

@author: Christian Konstantinov
"""

from functools import wraps
from time import time

def timed(f):
    """Measure the time it takes for f to execute."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('Elapsed time: {} ms'.format((end-start) * 1000))
        return result
    return wrapper
