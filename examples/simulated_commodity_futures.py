import numpy as np

import kalman_filter as kf

yt = np.genfromtxt('data/simulated_commodity_futures.csv',
                   delimiter=",")

yt = yt.transpose()

# yt = yt[:, :10]
# yt.fill(1)

# Develop a two-state variable, multi-measurement, time-varying measurement payload for the `kalman-filter` module:

# Total measurements, maximum measurements per measurement point:
d, n = yt.shape
# Total state variables:
m = 2
# Input arguments must adhere to these dimesions.

x = [3.1307, 0]
P = [[100, 0], [0, 100]]
dt = [-0.000236, 0]
Tt = [[1, 0], [0, 0.972]]
HHt = [[0.0003966981, 0.000231467], [0.0002314670, 0.001500735]]
ct = np.ndarray((d, n))
GGt = np.ndarray((d, n))
Zt = np.ndarray((d, m, n))

# Set dynamic time-varying arrays to constants for example purposes:
GGt[:] = 0.0016
ct[:] = -0.03
Zt[:] = 1

# Coerce inputs into a compatible `KalmanFilter` object with algorithm safety:
kalman_filter_input = kf.KalmanFilter(
    yt=yt, x=x, P=P, dt=dt, ct=ct, Tt=Tt, Zt=Zt, HHt=HHt, GGt=GGt)
print(kalman_filter_input)

# Given validated inputs for the Kalman filter algorithm, execute the algorithm:
kalman_filter_log_likelihood = kf.kalman_filter(kalman_filter_input)
print(kalman_filter_log_likelihood)

# For more involved algorithm outputs, execute the verbose filter:
kalman_filter_object = kf.kalman_filter_verbose(kalman_filter_input)
print(kalman_filter_object)

# Perform Kalman filtering, then smoothing:
kalman_filter_smoother = kf.kalman_smoother(kalman_filter_input)
print(kalman_filter_smoother)

# Kalman Smoothing of the filtered object:
kalman_filtered_smoothed = kf.kalman_smoother(kalman_filter_object)
print(kalman_filtered_smoothed)
kalman_filtered_smoothed.to_dict()
