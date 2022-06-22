# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 22:18:33 2022

@author: Christian Konstantinov
"""

import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple

class Filter(ABC):

    def __init__(self):
        self.filter = self._generate()

    @abstractmethod
    def _generate(self) -> np.ndarray:
        pass

    @abstractmethod
    def apply(self, input_sig: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, input_sig: np.ndarray) -> np.ndarray:
        return self.apply(input_sig)

class Decorrelator():

    def __init__(self, filters: Tuple[Tuple[Filter]], num_ins: int = 2, num_outs: int = 2):
        if len(filters) != num_outs:
            raise ValueError('Number of filters does not match number of output channels.')
        self.filters = filters
        self.num_ins = num_ins
        self.num_outs = num_outs

    def decorrelate(self, input_sig: np.ndarray) -> np.ndarray:
        output_sig = input_sig
        for ch in self.num_outs:
            for filt in self.filters:
                (cascaded_sig := filt[ch](input_sig[:, ch]))
            output_sig[:, ch] = cascaded_sig
        return output_sig

    def __call__(self, input_sig: np.ndarray) -> np.ndarray:
        return self.decorrelate(input_sig)
