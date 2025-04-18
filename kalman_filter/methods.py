from numpy import ndarray

from .models import KalmanFilter, KalmanFiltered, KalmanSmoothed
from .src import kalman_filter as KF


def kalman_filter(filter: KalmanFilter, verbose: bool = False) -> float:
    """Perform an optimized Kalman filter to calculate the maximum likelihoood estimate of current inputs against observations"""
    filter_input = filter.to_dict()
    if not verbose:
        # Call compiled C-Code with validated inputs:
        return KF.kalman_filter(filter_input)
    else:
        # Call compiled C-Code with validated inputs:
        output = KF.kalman_filter_verbose(filter_input)
        # Return validated outputs:
        return KalmanFiltered.from_dict(output | filter_input)


def kalman_filter_optimise(filter: dict[str, ndarray]) -> float:
    """Given unsafe inputs (as a dictionary), perform Kalman filtering to calculate the maximum likelihoood estimate of current inputs against observations.
    This function is designed for optimisation routines, reducing type safety overhead to deliver optimum performance.

    Unless you are familiar with this module and how inputs should be coerced / passed to this function, it is
      **highly recommended** you familiarise yourself with the `KalmanFilter` object and the `kalman_filter` functions.
    """
    return KF.kalman_filter(filter)


def kalman_smoother(filter: KalmanFilter | KalmanFiltered) -> KalmanSmoothed:
    filter_input = filter.to_dict()
    # Instance requires filtering, and then smoothing, values:
    if isinstance(filter, KalmanFilter):
        output = KF.kalman_filter_smoother(filter_input)
    # Instance has been filtered, only smoothing required:
    elif isinstance(filter, KalmanFiltered):
        output = KF.kalman_smoother(filter_input)
    return KalmanSmoothed.from_dict(output | filter_input)


def kalman_smoother_optimise(filter: dict[str, ndarray], filtered: bool = False):
    if filtered:
        return KF.kalman_smoother(filter)
    else:
        return KF.kalman_filter_smoother(filter)
