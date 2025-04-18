from numpy import ndarray

from .models import KalmanFilter, KalmanFiltered, KalmanSmoothed
from .src import kalman_filter as KF


def kalman_filter(filter: KalmanFilter) -> float:
    """Perform an optimized Kalman filter to calculate the maximum likelihoood estimate of current inputs against observations"""
    filter_input = filter.to_dict()
    # Call compiled C-Code with validated inputs:
    return KF.kalman_filter(filter_input)


def kalman_filter_optimise(filter: dict[str, ndarray]) -> float:
    """Given unsafe inputs (as a dictionary), perform Kalman filtering to calculate the maximum likelihoood estimate of current inputs against observations.
    This function is designed for optimisation routines, reducing type safety overhead to deliver optimum performance.

    Unless you are familiar with this module and how inputs should be coerced / passed to this function, it is
      **highly recommended** you familiarise yourself with the `KalmanFilter` object and the `kalman_filter` functions.
    """
    return KF.kalman_filter(filter)


def kalman_filter_verbose(filter: KalmanFilter) -> KalmanFiltered:
    """Perform an optimized Kalman filter to calculate the maximum likelihoood estimate of current inputs against observations"""
    filter_input = filter.to_dict()
    # Call compiled C-Code with validated inputs:
    output = KF.kalman_filter_verbose(filter_input)
    # Return validated outputs:
    return KalmanFiltered.from_dict(output | filter_input)


def kalman_smoother(smoother: KalmanFilter | KalmanFiltered) -> KalmanSmoothed:
    # Instance requires filtering, and then smoothing, values:
    if isinstance(smoother, KalmanFilter):
        filter_input = smoother.to_dict()
        filter_output = KF.kalman_filter_verbose(filter_input)
        smoother_input = filter_input | filter_output
    # Instance has been filtered, only smoothing required:
    elif isinstance(smoother, KalmanFiltered):
        smoother_input = filter_input | smoother.to_dict()
    else:
        raise TypeError(
            "arg: `filter` is not type `KalmanFilter` or `KalmanFiltered`!")

    # Call compiled C-Code with validated inputs:
    output = KF.kalman_smoother(smoother_input)
    return KalmanSmoothed.from_dict(output | filter_input)


def kalman_smoother_optimise(smoother: dict[str, ndarray], filtered: bool = False) -> dict[str, float | ndarray]:
    if filtered:
        smoother_input = smoother
    else:
        filter_output = KF.kalman_filter_verbose(smoother)
        smoother_input = smoother | filter_output
    return KF.kalman_smoother(smoother_input)
