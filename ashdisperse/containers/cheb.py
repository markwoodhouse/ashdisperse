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
# stored level-first, (chebIts, maxN, maxN), so that each level's N x N block
# is contiguous row by row
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
        Tn (float[:,:,:]): Array such that Tn[k, :N[k], :N[k]] contains the
                           chebyshev polynomials evaluated at the collocation
                           points for iterate k.
        dTn (float[:,:,:]): Array such that dTn[k, :N[k], :N[k]] contains the
                            derivative of the chebyshev polynomials evaluated
                            at the collocation points for iterate k.
        d2Tn (float[:,:,:]): Array such that d2Tn[k, :N[k], :N[k]] contains the
                             second derivative of the chebyshev polynomials
                             evaluated at the collocation points for iterate k.
    """

    def __init__(self, parameters):
        chebIts = parameters.solver.chebIts
        maxN = parameters.solver.maxN
        self.N = np.zeros((chebIts), dtype=np.int64)
        self.x = np.zeros((maxN, chebIts), dtype=np.float64)
        self.Tn = np.zeros((chebIts, maxN, maxN), dtype=np.float64)
        self.dTn = np.zeros((chebIts, maxN, maxN), dtype=np.float64)
        self.d2Tn = np.zeros((chebIts, maxN, maxN), dtype=np.float64)

        minN_log2 = parameters.solver.minN_log2
        for k in range(0, chebIts):
            N = 2**(k+minN_log2)
            self.N[k] = np.int64(N)
            x, T, dT, d2T = cheb(N)
            self.x[:N, k] = x
            self.Tn[k, :N, :N] = T
            self.dTn[k, :N, :N] = dT
            self.d2Tn[k, :N, :N] = d2T

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
        Tn = self.Tn[k, :N, :N]
        dTn = self.dTn[k, :N, :N]
        d2Tn = self.d2Tn[k, :N, :N]
        return x, Tn, dTn, d2Tn


# pylint: disable=E1101
ChebContainer_type = ChebContainer.class_type.instance_type
ChebContainer_type = ChebContainer.class_type.instance_type
