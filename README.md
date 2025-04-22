# Kalman-Filtering and Smoothing through optimised algorithms:

An implementation of the Kalman filtering and smoothing algorithm. This module is designed to be computationally efficient, using the sequential processing algorithm to increase computational efficiency. Designed specifically for those who have developed the underlying arrays that are used for computing the Kalman filter, but require this algorithm within numeric optimisation or within additional filtering/smoothing outputs.

## Quick Use:

Construct the input arrays of the Kalman filter and provide these to the module:

```

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

import kalman_filter
# Enforce algorithm safety using the `KalmanFilter` object:
kalman_filter_input = kalman_filter.KalmanFilter(**kalman_filter_dict)
print(kalman_filter_input)

# Given type-safety, execute the Kalman filter algorithm:
kalman_filter_log_likelihood = kalman_filter.kalman_filter(kalman_filter_input)
# Or execute the algorithm, returning output arrays:
kalman_filter_log_verbose = kalman_filter.kalman_filter_verbose(kalman_filter_input)
print(kalman_filter_log_likelihood)
````

## Additional Resources:

For those wishing to learn more about filtering and the linear Kalman filter, I highly recommend the online available textbook:  [Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)

### Why choose this module?

This module optimizes the application of the linear Kalman filter through two approaches:
- It utilizes a format of the algorithm called sequential processing.
- It uses the `numpy-C api` to execute the underlying algorithm using compiled C code

# Resources:

- Explore how Kalman filters work [using this interactive module](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
- Explore the distinction between traditional Kalman filtering and the sequential processing algorithm, and the benefits of the implemented sequential processing algorithm [pages 97-108](https://pure.bond.edu.au/ws/portalfiles/portal/167739220/Thomas_Aspinall_Thesis.pdf)
