from typing import Union

import numpy as np

# Compiled C code library:
import kalman_filter


# Attempting to Smooth before Kalman filter executed:
class KalmanFilterNotExecuted(Exception):
    pass


class OutOfRangeYt(Exception):
    pass


class KalmanFilter():

    def __init__(self,
                 yt: Union[float, list, np.ndarray],
                 x: Union[float, list, np.ndarray],
                 P: Union[float, list, np.ndarray],
                 dt: Union[float, list, np.ndarray],
                 ct: Union[float, list, np.ndarray],
                 Tt: Union[float, list, np.ndarray],
                 Zt: Union[float, list, np.ndarray],
                 HHt: Union[float, list, np.ndarray],
                 GGt: Union[float, list, np.ndarray],
                 ) -> None:

        # Instantiate variables -> to make them available to IDE's
        self.yt = yt
        self.x = x
        self.P = P
        self.dt = dt
        self.ct = ct
        self.Tt = Tt
        self.Zt = Zt
        self.HHt = HHt
        self.GGt = GGt

        # Input coercion, for scalar, iterables, etc.
        input_names = ["x", "P", "dt", "ct", "Tt", "Zt", "HHt", "GGt"]
        ndims = [1, 2, 2, 2, 3, 3, 3, 2]
        inputs = [x, P, dt, ct, Tt, Zt, HHt, GGt]
        for ndim, input, input_name in zip(ndims, inputs, input_names):
            # Scalar input support (making implicit assumption that d = 1):
            if np.isscalar(input):
                input_ndim = [1] * ndim
                input_ndarr = np.ndarray(input_ndim)
                input_ndarr[:] = input
            else:
                # sequence-like support:
                input_ndarr = np.array(input)
            setattr(self, input_name, input_ndarr)
        # yt handling:
        # Scalar input support (making implicit assumption that d = 1):
        if np.isscalar(yt):
            input_ndim = (1, len(yt))
            input_ndarr = np.ndarray(input_ndim)
            input_ndarr[:] = yt
            self.yt = input_ndarr
        else:
            # Scalar input support:
            yt_attr = np.array(yt)
            if yt_attr.ndim == 1:
                # Yt obtains its shape:
                ncol_yt = len(yt_attr)
                input_ndim = (1, ncol_yt)
                input_yt = np.ndarray(input_ndim)
                input_yt[0, :] = yt_attr
                # yt sequence-like support, which required column basis upon d:
                self.yt = input_yt
            else:
                OutOfRangeYt(
                    "yt must be either scalar, or a 1- or 2-dimensional array-like!")

        # Further Input dimension validation is performed within C calls:

        # Instantiate variables assigned during Kalman Smoothing:
        self.filtered = False
        self.smoothed = False
        self.v = None
        self.Kt = None
        self.Ft_inv = None
        self.xt = None
        self.Pt = None

    def _generate_filter_dict(self) -> dict[str, np.ndarray]:
        """Generates the input dictionary required of compiled C functions kalman_filter and kalman_filter_verbose. 
        Only typically intended to be called internally by the KalmanFilter class object. """
        return {
            "x": self.x,
            "P": self.P,
            "dt": self.dt,
            "ct": self.ct,
            "Tt": self.Tt,
            "Zt": self.Zt,
            "HHt": self.HHt,
            "GGt": self.GGt,
            "yt": self.yt
        }

    def _generate_smoother_dict(self) -> dict[str: np.ndarray]:
        return {
            "Tt": self.Tt,
            "Zt": self.Zt,
            "yt": self.yt,
            "v": self.v,
            "Kt": self.Kt,
            "Ft_inv": self.Ft_inv,
            "xt": self.xt,
            "Pt": self.Pt,
        }

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
