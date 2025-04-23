# Kalman-Filtering and Smoothing through optimised algorithms:

An implementation of the linear Kalman filtering and smoothing algorithms. This module is designed to be computationally efficient, implementing the "sequential processing" algorithm using the numpy C-api to increase algorithm performance. Designed specifically for those who have developed the underlying arrays that are used for computing the Kalman filter, but require this algorithm within numeric optimisation or within additional filtering/smoothing outputs.

# Quick Use:

Construct the input arrays of the Kalman filter and provide these to the module:

```python
import kalman_filter

kalman_filter_dict = {
    "yt": [200, 300, 400, 500],
    "x": 200,
    "P": 10000,
    "dt": 0,
    "ct": 0,
    "Tt": 1,
    "Zt": 1,
    "GGt": 15000,
    "HHt": 1300
}

# Enforce algorithm safety using the `KalmanFilter` object:
kalman_filter_input = kalman_filter.KalmanFilter.from_dict(kalman_filter_dict)

# Given type-safety, execute the Kalman filter algorithm:
log_likelihood = kalman_filter.kalman_filter(kalman_filter_input)

# Execute the algorithm and returning algorithm arrays:
filtered_outputs = kalman_filter.kalman_filter_verbose(kalman_filter_input)

# Execute the Kalman smoother algorithm:
smoothed_outputs = kalman_filter.kalman_smoother(kalman_filter_input)
```

# Sequential Procesing:

Traditional Kalman filtering takes the entire vector of measurements at each discrete measurement point for analysis. Sequential processing instead assumes that each measurment is independent and iterates over each measurment individually. This can be faster overall, as it requires less matrix operations (such as matrix inversions) and scales linearly with the total number of measurements.

The only difference in inputs between the traditional Kalman filter and sequential processing algorithms is the input for the error in measurements: `HHt`. Because measurements must be independent, it is not supported to have covariances in the disturbance of measurements through this algorithm.

# Additional Resources:

- Explore Kalman filtering [using this interactive module](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
- [Time Series Analysis by State Space Methods (2001)](https://academic.oup.com/book/16563) by Durbin and Koopman describes the sequential processing algorithm in greater detail.
- Explore the distinction between traditional Kalman filtering and the sequential processing algorithm, and the benefits in performance of the implemented sequential processing algorithm [pages 97-108](https://pure.bond.edu.au/ws/portalfiles/portal/167739220/Thomas_Aspinall_Thesis.pdf)
