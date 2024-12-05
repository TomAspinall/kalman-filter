# Linear Kalman Filtering and Smoothing:

A Fast and efficient implementation of the linear Kalman Filter and Smoother algorithms callable within python. When the efficiency of filtering, smoothing, and numeric parameter estimation of models relying on a linear Kalman filter process are crucial to applying your models.

For those wishing to learn more about filtering and the linear Kalman filter, I highly recommend the online available textbook: (Kalman-and-Bayesian-Filters-in-Python)[https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python]

This module is designed for those whom wish to:
- understand how a Kalman filter / smoother works, and,
- Estimate parameters of models following a linear Kalman filter through numeric optimisation (i.e., maximum likelihood estimate (MLE)), and
- Optimise the performance of Kalman filtering / smoothing over time-series data within Python.

## What is sequential processing?

Sequential processing is a method of linear Kalman filtering that, rather than obtaining matrices of measurements and their errors at each discrete time point, instead iteratively updates state variables as scalar values. (Worked examples to come).

This is a process that is:
- More computationally efficient than the "traditional" linear Kalman filter algorithm
- Scales linearly, as opposed to quadratically/exponentially, with respect to the dimensionality of yt i.e., the number of observations at each observation (Kindly see page 104 of my Ph.D. thesis)[https://pure.bond.edu.au/ws/portalfiles/portal/167739220/Thomas_Aspinall_Thesis.pdf]

If you have a singular observation at each discrete time-point, you won't even notice the difference, but will benefit from the gains of sequential processing.

If you have multiple observations at each discrete time point, the only additional restriction is that the error of measurements (denoted within this package as hht) must be independent. Even then, the Cholesky-Decomposition makes the pre-processing of such use cases trivial.

### I'm only measuring 1-2 variables with my sensor at any particular time-point, how does sequential processing help me?

- Even with a singular observation at each discrete time point i.e., a single dimension to your observations yt, sequential processing benefits from increased computational performance of 15-20% by applying sequential processing

## Why choose this module?

This module optimizes the application of the linear Kalman filter through two approaches:
- It utilizes a format of the algorithm called sequential processing. 
- It uses the numpy-C api to execute the underlying algorithm using compiled C code,

as a huge fan of the Kalman filter, I completely understand that existing textbooks focus on the derivation of this algorithm. However, in application, I have noticed that the Kalman filter may be utilised for numerical optimisation (when estimating the parameters of underlying models to fit your observations). This means the Kalman filter algorithm may be applied to training data potentially 100,000's of times. This makes optimisation important.

Utilising numerical optimisation to estimate parameters can be a computationally expensive task. This is purpose of this module.


# Resources:

- Explore how Kalman filters work (using this interactive module)[https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python]
- (Shamelesslty plugged) explore the disctinction between traditional Kalman filtering and the sequential processing algorithm within my PhD (between page 97 and 108)[https://pure.bond.edu.au/ws/portalfiles/portal/167739220/Thomas_Aspinall_Thesis.pdf]
