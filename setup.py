from distutils.core import Extension, setup

import numpy as np

module = Extension("kalman_filter", sources=[
                   "kalman_filter_sp.c"])

setup(
    name="example",
    version="0.0.1",
    description="Efficient Kalman Filtering and Smoothing through the sequential processing method",
    ext_modules=[module],
    include_dirs=[np.get_include()]
)

# python setup.py build
