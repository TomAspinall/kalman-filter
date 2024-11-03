from distutils.core import Extension, setup

import numpy as np

# cblas:
cblas_include_dir = "C:\\Users\\Tpm Aspinall\\Documents\\Work\\OpenBLAS-0.3.28-x64-64\\include\\"
cblas_library_dir = "C:\\Users\\Tpm Aspinall\\Documents\\Work\\OpenBLAS-0.3.28-x64-64\\lib\\"
numpy_include_dir = np.get_include()
include_dirs = [numpy_include_dir, cblas_include_dir]
library_dirs = [cblas_library_dir]

c_sources = ["kalman_filter_sp.c"]

module = Extension("kalman_filter", sources=c_sources,
                   libraries=["libopenblas"],
                   include_dirs=include_dirs,
                   library_dirs=[cblas_library_dir])

setup(
    name="example",
    version="0.0.1",
    description="Efficient Kalman Filtering and Smoothing through the sequential processing method",
    ext_modules=[module],
    include_dirs=include_dirs
)

# python setup.py build
