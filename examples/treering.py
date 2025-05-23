import json

import kalman_filter as kf

# Load example data:
with open('data/treering.json', 'r') as f:
    treering = json.load(f)

# Develop a simplified one-factor, one-measurement, time constant measurement payload for the `kalman-filter` module:

# Total measurements:
n = len(treering)
# Measurements per time point:
d = 1
# Total state variables:
m = 1
# Input arguments must adhere to these dimesions.

# Establish input arrays and their dimensions:
# Noting that iterables and floats are coerced into the correct Kalman filter dimensions:
yt = treering
# Estimation of the first year flow:
x = yt[0]
# Little is known of the initial estimate. Setting a 'diffuse' estimate here:
P = 10000
# Set transition and measurement equation arguments:
dt = 0
ct = 0
Tt = 1
Zt = 1
# Little is known of these arguments initially, setting them to very high values:
# Note: these arguments could possibly be estimated through MLE:
GGt = 15000
HHt = 1300

# Transform inputs into a coercable:
kalman_filter_dict = {
    "yt": yt,
    "x": x,
    "P": P,
    "dt": dt,
    "ct": ct,
    "Tt": Tt,
    "Zt": Zt,
    "GGt": GGt,
    "HHt": HHt
}

# Coerce inputs into a compatible `KalmanFilter` object with algorithm safety:
kalman_filter_input = kf.KalmanFilter(**kalman_filter_dict)
print(kalman_filter_input)

# Given validated inputs for the Kalman filter algorithm, execute the algorithm:
kalman_filter_log_likelihood = kf.kalman_filter(kalman_filter_input)
print(kalman_filter_log_likelihood)
# For more involved algorithm outputs, execute the verbose filter:
kalman_filter_object = kf.kalman_filter_verbose(kalman_filter_input)
print(kalman_filter_object)

# Kalman Smoothing of the filtered object:
kalman_filtered_smoothed = kf.kalman_smoother(kalman_filter_object)
print(kalman_filtered_smoothed)

# Perform Kalman filtering, then smoothing:
kalman_filter_smoother = kf.kalman_smoother(kalman_filter_input)
print(kalman_filter_smoother)
