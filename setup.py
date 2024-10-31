from distutils.core import Extension, setup

module = Extension("kalman_filter", sources=["kalman_filter_sp.c"])

setup(
    name="example",
    version="0.0.1",
    description="Efficient Kalman Filtering and Smoothing through the sequential processing method",
    ext_modules=[module])

# python setup.py build
