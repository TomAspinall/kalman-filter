import os
from distutils.core import Extension, setup

import numpy as np

# Compiled C code directory:
src_directory = os.path.join("kalman_filter", "c_core")

CBLAS_DIR = os.getenv("CBLAS_DIR", os.path.join(
    "external", "OpenBLAS-0.3.28-x64-64"))
cblas_include_dir = os.path.join(CBLAS_DIR, "include")
cblas_library_dir = os.path.join(CBLAS_DIR, "lib")

# cblas:
numpy_include_dir = np.get_include()

c_sources = [os.path.join(src_directory, x)
             for x in os.listdir(src_directory) if x.endswith('.c')]

module = Extension("kalman_filter.c_core", sources=c_sources,
                   libraries=["libopenblas"],
                   include_dirs=[numpy_include_dir,
                                 cblas_include_dir, src_directory, "."],
                   define_macros=[
                       ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                   library_dirs=[cblas_library_dir])

setup(
    ext_modules=[module],
)
