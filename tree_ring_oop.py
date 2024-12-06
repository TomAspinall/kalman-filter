import numpy as np

__name__ = "kalman_filter.test"

import json

import kalman_filter_object

# with open('test/tree_ring.csv', 'r') as f:
#     tree_ring = f.read()
# values = [x.split(',')[-1] for x in tree_ring.split('\n')]
# vals = [float(x) for x in values if x not in ('', '"x"')]
# with open('test/tree_ring.json', 'w') as f:
#     json.dump(vals, f, indent=1)

with open('test/tree_ring.json', 'r') as f:
    tree_ring = json.load(f)

# Total observations:
n = len(tree_ring)
# Observations per time point:
d = 1
# State variables:
m = 1
# Constant i.e., not time-varying:
const = True


yt = tree_ring

KF = kalman_filter_object.KalmanFilter(
    yt=np.array(tree_ring),
    x=tree_ring[0],  # Estimation of the first year flow:
    P=100,
    dt=0,
    ct=0,
    Zt=1,
    Tt=1,
    GGt=0.0822359114,
    HHt=0.0004871744
)

filtered_ll = KF.kalman_filter()
filtered = KF.kalman_filter_verbose()

for key, value in KF._generate_smoother_dict().items():
    print(key, value.shape)


print("Log Likelihood:", filtered_ll)
print("Log Likelihood (Verbose):", filtered['log_likelihood'])
# -1666.0949064512643

smoothed = KF.kalman_smoother()
# print("Vt:", smoothed["Vt"][0, 0, :])
# print("xhatt", smoothed["xhatt"][0, :])
filtered_smoothed = KF.kalman_filter_smoother()

print("Vt:", smoothed.Vt[0, 0, 0])
print("xhatt", smoothed.xhatt[0, 0])

print('')

# Vt: 97.44471825622102
# xhatt 1119.7736885015963
