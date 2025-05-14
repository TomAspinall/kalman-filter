from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .base import BaseClassExtended
from .exceptions import InputOutOfRange, ShapeIncompatible


@dataclass
class KalmanFilter(BaseClassExtended):
    """Kalman Filter compatible object"""
    yt: float | Iterable | np.ndarray
    """`yt`: Observed measurements at each measurement point. N/A's are allowed for missing measurements, with individual measurements skipped at each missing measurement point.
    
    Array dimensions:
     - 2-dimensional `(d, n)` - A `d` number of measurements observed at an `n` number of measurement points.

     Subsequent `KalmanFilter` arguments / algorithm objects must conform to the size of the first and second dimensions of object `yt`: `(d, n)`.
    """
    x: float | Iterable | np.ndarray
    """`x`: initial value / estimation of the state variable(s)
    
    Array dimensions:
     - 1-dimensional `(m, 1)`
    """
    P: float | Iterable | np.ndarray
    """`P`: initial variance of state variables `x`. 
    
    When little is known about the initial variance of state variables, it is common to instatiate with large initial diagonal values (diffusion variance matrix).
    """
    dt: float | Iterable | np.ndarray
    """`dt`: intercept of the transition equation.
    
    Available array dimensions:
     - 1-dimensional: `(m, 1)` - constant intercept for each transition between measurement points.
     - 2-dimensional: `(m, n)` - time-varying intercept for each transition between measurement points.
    """
    ct: float | Iterable | np.ndarray
    """`ct`: intercept of the measurement equation.
    
    Available array dimensions:
     - 1-dimensional: `(d, 1)` - constant intercept to translate state variables into each measurement.
     - 2-dimensional: `(d, n)` - time-varying intercept to translate state variables into each measurement at each measurement point.
    """
    Tt: float | Iterable | np.ndarray
    """`Tt`: factor of the transition equation. (i.e., the "slope" of the transition equation)
    
    Available array dimensions:
     - 2-dimensional: `(m, m)` - constant factor for each state variable.
     - 3-dimensional: `(m, m, d)` - Time-varying factor for each state variable at each measurement point.
    """
    Zt: float | Iterable | np.ndarray
    """`Zt`: factor of the measurement equation. (i.e., the "slope" of the measurement equation)
    
    Available array dimensions:
     - 2-dimensional: `(d, m)` - Constant factor to translate state variable(s) into each measurement.
     - 3-dimensional: `(d, m, n)` - Time-varying factor to translate state variable(s) into each measurement at each measurement point.
    
    """
    HHt: float | Iterable | np.ndarray
    """`HHt`: variance of the innovations of the transition equation. i.e., "white noise" in the development of state variables over time.
    
    Available array dimensions:
     - 2-dimensional: `(m, m)` - Constant variance for the innovations of state variables between measurement points.
     - 3-dimensional: `(m, m, n)` - Time-varying variance for the innovations of state variables between measurement points.
    """
    GGt: float | Iterable | np.ndarray
    """`GGt`: Diagonal elements of a matrix for the variance of disturbances on the measurement equation. i.e., "white noise" in measurements.

    Available array dimensions:
     - 1-dimensional: `(d, 1)` - Constant disturbances for each measurement at each measurement point.
     - 2-dimensional: `(d, n)` - Time-varying disturbances for each measurement at each measurement point.
    
    Covariance betwen disturbances is not supported. (i.e., multiple measurements at each discrete measurement point cannot have covariance in white noise). The sequential processing algorithm (the algorithm used in the `kalman-filter` module) makes the explicit assumption
    that these observations / measurements are independent, which results in a significantly faster filtering algorithm.

    Several classic techniques to develop a diagonal covariance matrix may assist in coercing values of `HHt` that meet this dependency.
    
    """

    def __post_init__(self):
        """Coerce input attributes, enforcing compatible dtypes and dimensions (where applicable)"""

        # Enforced ndims for inputs:
        self._attr_expected_ndims = {
            "x": 1,
            "P": 2,
            "dt": 2,
            "ct": 2,
            "Tt": 3,
            "Zt": 3,
            "HHt": 3,
            "GGt": 2
        }
        for attr, ndim in self._attr_expected_ndims.items():
            input_ndarr = getattr(self, attr)
            # Enforce np.ndarray:
            input_ndarr = np.array(input_ndarr, dtype="float64")
            # Enforce shape to match number of dimensions:
            for _ in range(ndim - input_ndarr.ndim):
                input_ndarr = np.expand_dims(
                    input_ndarr, axis=input_ndarr.ndim)
            setattr(self, attr, input_ndarr)

        # yt coercion:
        # Scalar input support (making implicit assumption that d = 1):
        yt_attr = np.array(self.yt, dtype="float64")
        if yt_attr.ndim == 0:
            yt_attr = yt_attr.reshape(shape=(1, 1))
        # Yt must be a column vector:
        elif yt_attr.ndim == 1:
            yt_attr = yt_attr.reshape((1, len(yt_attr)))
        elif yt_attr.ndim > 2:
            raise InputOutOfRange(
                "yt must be either scalar, or a 1- or 2-dimensional array-like!")
        self.yt = yt_attr

        # Enforce Kalman filter dimensions:
        self._input_dimension_checks()

    def _input_dimension_checks(self):
        # Total state variables:
        m = self.x.shape[0]
        # Total measurements at each measurement point:
        d = self.yt.shape[0]
        # Total measurement points:
        n = self.yt.shape[1] if self.yt.ndim > 1 else 1
        if any(x != m for x in self.P.shape):
            raise ShapeIncompatible(
                "`P` - `(m, m)` dimensions do not match first dimension of `x` - `(m, 1)`")
        elif self.dt.shape[0] != m:
            raise ShapeIncompatible(
                "`dt` - `(m, d)` dimensions do not match `x` - `(m, 1)`")
        elif self.dt.ndim > 1 and self.dt.shape[1] not in (1, d):
            raise ShapeIncompatible(
                "`dt` - `(m, d)` dimensions do not match `yt` - `(d, n)`")
        elif any(x != m for x in self.Tt.shape[:1]):
            if self.Tt.shape[2] == 1:
                raise ShapeIncompatible(
                    "`Tt` - `(m, m)` dimensions do not match `x` - `(m, 1)`")
            else:
                raise ShapeIncompatible(
                    "`Tt` - `(m, m, d)` dimensions do not match `x` - `(m, 1)`")
        elif self.Tt.ndim > 2 and self.Tt.shape[2] not in (1, d):
            raise ShapeIncompatible(
                "`Tt` - `(m, m, d)` dimensions do not match `yt` - `(d, n)`")

    # Print condensed dimensions rather than arrays, which may be verbose:
    def __repr__(self) -> str:
        return f"KalmanFilter(yt={self.yt.shape}, x={self.x.shape}, P={self.P.shape}, dt={self.dt.shape}, ct={self.ct.shape}, Tt={self.Tt.shape}, Zt={self.Zt.shape}, HHt={self.HHt.shape}, GGt={self.GGt.shape})"


@dataclass
class KalmanFiltered(KalmanFilter):
    """
    Outputs of the Kalman filter through the sequential processing algorithm, containing a time series of estimated states, covariances, and associated outputs.

    attributes:
     - log-likelihoods
     - state estimates
     - predicted measurements
     - Kalman gain
     - Prediction error variance matrix.
    """
    log_likelihood: float
    """`log_likelihood`: calculated log-likelihood of observed state variables fit to measurements. This log-likelihood quantifies the goodness of fit between the models' predicted states and the measured observations, based on the input parameters.
    """
    vt: np.ndarray
    """Known as the innovation prediction error."""
    Kt: np.ndarray
    """`Kt`: Kalman gain. Used within the Kalman filter to update the state estimate with new measurements. `Kt` balances the uncertainty in the predicted state with the uncertainty in the observation, determining how much the measurements `yt` influcence the state update between measurement points.
    
    Array dimensions:
     - `(m, m, n)`
    """
    Ft_inv: np.ndarray
    """The inverse of the prediction error variance matrix. 
    
    The values of `Ft_inv` may differ to traditional (matrices based) Kalman filter algorithms, 
    as this module instead utilises the sequential processing method for computational efficiencies.

    Only `Ft_inv[0,0]` will be identical to traditional, matrices based algorithms.

    Array dimensions:
     - `(m, n)`
    '"""
    xtt: np.ndarray
    """
    `xtt`: filtered state variables.

    Array dimensions:
     - `(m, n)`
    """
    Ptt: np.ndarray
    """
    `Ptt`: variance / covariance matrix at each discrete measurement point of filtered state variables.

    Array dimensions:
     - `(m, m, n)`
    """

    # Print condensed dimensions rather than arrays, which may be verbose:
    def __repr__(self) -> str:
        return f"KalmanFiltered(log_likelihood= {self.log_likelihood:.4f}, vt={self.vt.shape}, Kt={self.Kt.shape}, Ft_inv={self.Ft_inv.shape}, xtt={self.xtt.shape}, Ptt={self.Ptt.shape})"


@dataclass
class KalmanSmoothed(KalmanFiltered):
    """Outputs of the Kalman filter/smoother through the sequential processing algorithm."""
    xhatt: np.ndarray
    """`xhatt`: smoothed state variables
    
    Array dimensions:
     - `(m, n)`
    """
    Vt: np.ndarray
    """`Vt`: variance / covariance matrix at each discrete measurement point of smoothed state variables.

    Array dimensions:
     - `(m, m, n)`
    """

    # Print condensed dimensions rather than arrays, which may be verbose:
    def __repr__(self) -> str:
        return f"KalmanSmoothed(log_likelihood= {self.log_likelihood:.4f}, xhatt={self.xhatt.shape}, Vt={self.Vt.shape})"
