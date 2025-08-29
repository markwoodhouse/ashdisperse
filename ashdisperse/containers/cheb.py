# -*- coding: utf-8 -*-
"""cheb.py

This module defines a container for Chebyshev polynomials, stored as a numba
jitclass.
"""

from collections import OrderedDict

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass

from ..spectral.cheb import cheb

# pylint: disable=C0103

Cheb_spec = OrderedDict()
Cheb_spec['N'] = int64[::1]       # degrees of sets of the polynomial
Cheb_spec['x'] = float64[:, ::1]  # collocation points as arrays
# Chebyshev polynomials
Cheb_spec['Tn'] = float64[:, :, ::1]
Cheb_spec['dTn'] = float64[:, :, ::1]   # first derivative
Cheb_spec['d2Tn'] = float64[:, :, ::1]  # second derivatives




@jitclass(Cheb_spec)
class ChebContainer():
    """A container for chebyshev matrices.

    ChebContainer contains chebyshev matrices for different degrees of
    approximation.

    Attributes:
        N (int[:]): Array of the degrees of the chebyshev approximations.
        x (float[:,:]): Array such that x[:N[k], k] contains the collocation
                        points for the chebyshev approximation at iterate k.
        Tn (float[:,:,:]): Array such that Tn[:N[k], :N[k], k] contains the
                           chebyshev polynomials evaluated at the collocation
                           points for iterate k.
        dTn (float[:,:,:]): Array such that dTn[:N[k], :N[k], k] contains the
                            derivative of the chebyshev polynomials evaluated
                            at the collocation points for iterate k.
        d2Tn (float[:,:,:]): Array such that d2Tn[:N[k], :N[k], k] contains the
                             second derivative of the chebyshev polynomials
                             evaluated at the collocation points for iterate k.
    """

    def __init__(self, parameters):
        chebIts = parameters.solver.chebIts
        maxN = parameters.solver.maxN
        self.N = np.zeros((chebIts), dtype=np.int64)
        self.x = np.zeros((maxN, chebIts), dtype=np.float64)
        self.Tn = np.zeros((maxN, maxN, chebIts), dtype=np.float64)
        self.dTn = np.zeros((maxN, maxN, chebIts), dtype=np.float64)
        self.d2Tn = np.zeros((maxN, maxN, chebIts), dtype=np.float64)

        minN_log2 = parameters.solver.minN_log2
        for k in range(0, chebIts):
            N = 2**(k+minN_log2)
            self.N[k] = np.int64(N)
            x, T, dT, d2T = cheb(N)
            self.x[:N, k] = x
            self.Tn[:N, :N, k] = T
            self.dTn[:N, :N, k] = dT
            self.d2Tn[:N, :N, k] = d2T

    def get_cheb(self, k):
        """Get Chebyshev matrices.

        This function gets chebyshev matrices from the ChebContainter.

        Args:
            k (int): the index of the iterates.

        Raises:
            ValueError: if the index is negative or greater than N.size
        """
        if k < 0 or k > self.N.size:
            raise ValueError(
                'In VelocityContainer upper_z, k must be in the range '
                + '[0, parameters.solver.chebIts].')
        N = self.N[k]
        x = self.x[:N, k]
        Tn = self.Tn[:N, :N, k]
        dTn = self.dTn[:N, :N, k]
        d2Tn = self.d2Tn[:N, :N, k]
        return x, Tn, dTn, d2Tn


# pylint: disable=E1101
ChebContainer_type = ChebContainer.class_type.instance_type
ChebContainer_type = ChebContainer.class_type.instance_type
