import os
from distutils.core import Extension, setup

import numpy as np

# Current path:
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

CBLAS_DIR = os.path.join(BASE_DIR, "external", "OpenBLAS-0.3.28-x64-64")
cblas_include_dir = os.path.join(CBLAS_DIR, "include")
cblas_library_dir = os.path.join(CBLAS_DIR, "lib")

# Compiled C code directory:
src_directory = os.path.join(BASE_DIR, "kalman_filter", "native")


# cblas:
cblas_include_dir = "C:\\Users\\Tpm Aspinall\\Documents\\Work\\OpenBLAS-0.3.28-x64-64\\include\\"
cblas_library_dir = "C:\\Users\\Tpm Aspinall\\Documents\\Work\\OpenBLAS-0.3.28-x64-64\\lib\\"
numpy_include_dir = np.get_include()
include_dirs = [numpy_include_dir, cblas_include_dir, src_directory]
library_dirs = [cblas_library_dir]

c_sources = ["kalman_filter_verbose.c", "utils.c",
             "init.c", "kalman_filter.c", "kalman_smoother.c"]
# Absolute paths:
c_sources = [os.path.join(src_directory, src) for src in c_sources]

module = Extension("kalman_filter", sources=c_sources,
                   libraries=["libopenblas"],
                   include_dirs=include_dirs,
                   library_dirs=[cblas_library_dir])

setup(
    name="kalman_filter",
    version="0.0.1",
    description="Efficient Traditional Kalman Filtering and Smoothing using the sequential processing algorithm",
    ext_modules=[module],
    include_dirs=include_dirs
)

# python setup.py build
