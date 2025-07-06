
class KalmanFilterError(Exception):
    """Base class for all kalman-filter module related exceptions."""
    pass

# A provided input to a dataclass is considered out of range:


class InputOutOfRange(KalmanFilterError):
    """Raised when the dimensions of `yt` are > 2, which is not supported for time-series analysis / Kalman filtering."""
    pass


class ShapeIncompatible(KalmanFilterError):
    """Raised when an input matrix has incompatible dimensions for input into the Kalman filter / smoother algorithm."""
    pass
