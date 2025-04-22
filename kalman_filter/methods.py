from numpy import ndarray

# from .native import c_core as KF
from . import c_core as KF
from .models import KalmanFilter, KalmanFiltered, KalmanSmoothed


def kalman_filter(filter: KalmanFilter) -> float:
    """Perform an optimized Kalman filter iterative algorithm to calculate the log-likelihood for the observed fit of the current input vector against measurments over a measurement period."""
    # Coerce inputs to the core algorithm:
    filter_input = filter.to_dict()
    # Call compiled C-Code with validated inputs:
    return KF.kalman_filter(filter_input)


def kalman_filter_optimise(filter: dict[str, ndarray]) -> float:
    """Unsafe variant of the `kalman_filter` method. Designed for optimisation routines, reducing type safety overhead to deliver optimum performance when run in an optimisation loop.

    It is **highly recommended* you use the `kalman_filter` function and `KalmanFilter` object if you are unfamiliar with the structure of your algorithm.
    """
    return KF.kalman_filter(filter)


def kalman_filter_verbose(filter: KalmanFilter) -> KalmanFiltered:
    """Perform an optimized Kalman filter iterative algorithm to calculate and return relevant algorithm arrays and matrices."""
    # Coerce inputs to the core algorithm:
    filter_input = filter.to_dict()
    # Call compiled C-Code with validated inputs:
    output = KF.kalman_filter_verbose(filter_input)
    # Return validated outputs:
    return KalmanFiltered.from_dict(output | filter_input)


def kalman_smoother(smoother: KalmanFilter | KalmanFiltered) -> KalmanSmoothed:
    """Perform an optimized Kalman filter iterative algorithm to calculate and return relevant algorithm arrays and matrices."""
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
    return KalmanSmoothed.from_dict(output | smoother_input)


def kalman_smoother_optimise(smoother: dict[str, ndarray], filtered: bool = False) -> dict[str, float | ndarray]:
    if filtered:
        smoother_input = smoother
    else:
        filter_output = KF.kalman_filter_verbose(smoother)
        smoother_input = smoother | filter_output
    return KF.kalman_smoother(smoother_input)
