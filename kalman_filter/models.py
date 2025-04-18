import inspect
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .exceptions import InputOutOfRange, ShapeIncompatible


@dataclass
class KalmanFilter():
    """Kalman Filter compatible object"""
    yt: float | Iterable | np.ndarray
    """`yt`: Observed measurements at each measurement point. N/A's are allowed for missing observations, and are skipped.
    
    Array dimensions:
     - 2-dimensional `(d, n)` - A `d` number of measurements observed across an `n` number of measurement points.

     Following Kalman filter algorithm objects must conform to the size of the first and second dimensions of object `yt`
    """
    x: float | Iterable | np.ndarray
    """`x`: the initial value / estimation of the state variable(s)
    
    Array dimensions:
     - 1-dimensional `(m, 1)`
    """
    P: float | Iterable | np.ndarray
    """`P`: the initial variance of state variables `x`. 
    
    When little is known about the initial variance of state variables, it is common to instatiate with large initial diagonal values (diffusion variance matrix).
    """
    dt: float | Iterable | np.ndarray
    """`dt`: The intercept of the transition equation.
    
    Available array dimensions:
     - 1-dimensional: `(m, 1)` - constant intercept for each transition between measurement points.
     - 2-dimensional: `(m, n)` - time-varying intercept for each transition between measurement points.
    """
    ct: float | Iterable | np.ndarray
    """`ct`: the intercept of the measurement equation.
    
    Available array dimensions:
     - 1-dimensional: `(d, 1)` - constant intercept to translate state variables into each measurement.
     - 2-dimensional: `(d, n)` - time-varying incercept to translate state variables into each measurement at each measurement point.
    """
    Tt: float | Iterable | np.ndarray
    """`Tt`: the factor of the transition equation.
    
    Available array dimensions:
     - 2-dimensional: `(m, m)` - constant factor for each state variable.
     - 3-dimensional: `(m, m, d)` - Time-varying factor for each state variable at each measurement point.
    """
    Zt: float | Iterable | np.ndarray
    """`Zt`: the multiplicative factor of the measurement equation.
    
    Available array dimensions:
     - 2-dimensional: `(d, m)` - Constant factor to translate state variable(s) into each measurement.
     - 3-dimensional: `(d, m, n)` - Time-varying factor to translate state variable(s) into each measurement at each measurement point.
    
    """
    HHt: float | Iterable | np.ndarray
    """`HHt`: the variance of the innovations of the transition equation. i.e., "white noise" in the development of state variables over time.
    
    Available array dimensions:
     - 2-dimensional: `(m, m)` - Constant variance for the innovations of state variables between measurement points.
     - 3-dimensional: `(m, m, n)` - Time-varying variance for the innovations of state variables between measurement points.
    """
    GGt: float | Iterable | np.ndarray
    """`GGt`: Diagonal elements of a matrix for the variance of disturbances on the measurement equation. i.e., "white noise" in measurements.

    Available array dimensions:
     - 1-dimensional: `(d, 1)` - Constant disturbances for each measurement at each measurement point.
     - 2-dimensional: `(d, n)` - Time-varying disturbances for each measurement at each measurement point.
    
    Covariance betwen disturbances is not supported using the sequential processing algorithm when there are multiple observations at each discrete time point. The sequential processing algorithm makes the explicit assumption
    that these observations / measurements are independent, which results in a significantly faster filtering algorithm.
    
    """

    def __post_init__(self):
        # Explicit n, m, d dimensional compatibility is enforced within the compiled C algorithm implementations.

        # Coerce inputs, checking for compatible dtypes:
        # attr, expected ndims:
        expected_ndims = {
            "x": 1,
            "P": 2,
            "dt": 2,
            "ct": 2,
            "Tt": 3,
            "Zt": 3,
            "HHt": 3,
            "GGt": 2
        }
        for attr, ndim in expected_ndims.items():
            input = getattr(self, attr)
            # Scalar input support:
            if np.isscalar(input):
                input_ndarr = np.ndarray([1] * ndim)
                input_ndarr[:] = input
            elif type(input) != np.ndarray:
                # Iterable support:
                input_ndarr = np.array(input)
            else:
                # No coercion necessary:
                continue
            setattr(self, attr, input_ndarr)

        # yt coercion:
        # Scalar input support (making implicit assumption that d = 1):
        if np.isscalar(self.yt):
            yt_attr = np.ndarray((1, 1))
            yt_attr[:] = self.yt
        elif type(self.yt) != np.ndarray:
            # Iterable input support:
            yt_attr = np.array(self.yt)
            if yt_attr.ndim == 1:
                # Yt must be transposed into a column vector:
                ncol_yt = len(yt_attr)
                input_ndim = (1, ncol_yt)
                input_yt = np.ndarray(input_ndim)
                input_yt[0, :] = yt_attr
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
        elif self.dt.shape[1] != 1 and self.dt.shape[1] != d:
            raise ShapeIncompatible(
                "`dt` - `(m, d)` dimensions do not match `yt` - `(d, n)`")
        elif any(x != m for x in self.Tt.shape[:1]):
            if self.Tt.shape[2] == 1:
                raise ShapeIncompatible(
                    "`Tt` - `(m, m)` dimensions do not match `x` - `(m, 1)`")
            else:
                raise ShapeIncompatible(
                    "`Tt` - `(m, m, d)` dimensions do not match `x` - `(m, 1)`")
        elif (self.Tt.shape[2] > 1 and self.Tt.shape[2] != d):
            raise ShapeIncompatible(
                "`Tt` - `(m, m, d)` dimensions do not match `yt` - `(d, n)`")

    # Make serialisable:

    def to_dict(self):
        """Return the coerced attributes of a KalmanFilter object as a dictionary"""
        return asdict(self)

    # Make subscriptable:
    def __getitem__(self, item):
        return getattr(self, item)

    # Build class from dict, ignoring additional kwargs:
    @classmethod
    def from_dict(cls, input):
        class_attributes = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in input.items() if k in class_attributes})

    # Print condensed dimensions rather than arrays, which may be verbose:
    def __repr__(self) -> str:
        return f"KalmanFilter(yt={self.yt.shape}, x={self.x.shape}, P={self.P.shape}, dt={self.dt.shape}, ct={self.ct.shape}, Tt={self.Tt.shape}, Zt={self.Zt.shape}, HHt={self.HHt.shape}, GGt={self.GGt.shape})"


@dataclass
class KalmanFiltered(KalmanFilter):
    """Kalman filter has been applied"""
    log_likelihood: float
    """The log-likelihood of observed state variables fit to observations. This is calculated within the Kalman filter algorithm by:
    
    log_likelihood = sum(t)^n
    """
    vt: np.ndarray
    Kt: np.ndarray
    Ft_inv: np.ndarray
    xtt: np.ndarray
    Ptt: np.ndarray

    # Print condensed dimensions rather than arrays, which may be verbose:
    def __repr__(self) -> str:
        return f"KalmanFiltered(log_likelihood={self.log_likelihood:,.4f}, vt={self.vt.shape}, Kt={self.Kt.shape}, Ft_inv={self.Ft_inv.shape}, xtt={self.xtt.shape}, Ptt={self.Ptt.shape})"


@dataclass
class KalmanSmoothed(KalmanFiltered):
    """Kalman filter and smoother has been applied"""
    xhatt: np.ndarray
    Vt: np.ndarray

    # Print condensed dimensions rather than arrays, which may be verbose:
    def __repr__(self) -> str:
        return f"KalmanSmoothed(log_likelihood={self.log_likelihood:,.4f}, xhatt={self.xhatt.shape}, Vt={self.Vt.shape}"
