from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass
class KalmanFilter():
    """Kalman Filter compatible object"""
    yt: float | Iterable | np.ndarray
    x: float | Iterable | np.ndarray
    P: float | Iterable | np.ndarray
    dt: float | Iterable | np.ndarray
    ct: float | Iterable | np.ndarray
    Tt: float | Iterable | np.ndarray
    Zt: float | Iterable | np.ndarray
    HHt: float | Iterable | np.ndarray
    GGt: float | Iterable | np.ndarray

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
            yt_attr = np.ndarray((1, len(self.yt)))
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

    # Make serialisable:
    def to_dict(self):
        """Return the coerced attributes of a KalmanFilter object as a dictionary"""
        return asdict(self)

    # Make subscriptable:
    def __getitem__(self, item):
        return getattr(self, item)

    def kalman_filter_verbose(self) -> dict[str, Union[float, np.ndarray]]:
        # Call C process:
        filtered_output = kalman_filter.kalman_filter_verbose(
            self._generate_filter_dict())
        # Assign filtered outputs to class variables:
        self.filtered = True
        self.v = filtered_output["vt"]
        self.Kt = filtered_output["Kt"]
        self.Ft_inv = filtered_output["Ft_inv"]
        self.xt = filtered_output["xtt"]
        self.Pt = filtered_output["Ptt"]

        return filtered_output

    def kalman_filter(self) -> float:
        # Call C process:
        return kalman_filter.kalman_filter(self._generate_filter_dict())

    def kalman_smoother(self) -> dict[str, np.ndarray]:
        """Requires that the kalman_filter class method has been executed."""
        if not self.filtered:
            # Filtered values not yet obtained. Run filter process, then smoother process:
            filtered_smoothed = self.kalman_filter_smoother(
                self._generate_smoother_dict())

        # Call C process:
        smoothed = kalman_filter.kalman_smoother(
            self._generate_smoother_dict())

        # Unpack result:
        self.smoothed = True
        self.xhatt = smoothed["xhatt"]
        self.Vt = smoothed["Vt"]
        return smoothed

    def kalman_filter_smoother(self) -> dict[str, Union[float, np.ndarray]]:
        """Consecutively executes both the linear forward Kalman filter algorithm and the backwards Kalman smoother algorithm 
        within the same compiled C call, which may benefit from reduced computational overhead. 
        The sequential processing algorithm is utilised for filtering / smoothing, which may result in inconsistencies in intermediate filtered values of ().
        See WhySequentialProcessing() for more detail."""
        pass


@dataclass
class KalmanFiltered(KalmanFilter):
    vt: str
