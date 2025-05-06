import os
import shutil
import sys
from distutils.core import Extension, setup

import numpy as np
import scipy_openblas64

# Compiled C code directory:
src_directory = os.path.join("kalman_filter", "c_core")

# openblas:
cblas_dir = os.path.dirname(scipy_openblas64.get_include_dir())
cblas_include_dir = scipy_openblas64.get_include_dir()
cblas_library_dir = scipy_openblas64.get_lib_dir()

# Optional: set runtime library search path (-rpath)
extra_link_args = [
    "-Wl,-rpath,@loader_path/../.dylibs"] if sys.platform == "darwin" else []

libraries = [scipy_openblas64.get_library()]

# Copy dll to make wheels self contained:
if os.name == "nt":
    dll_file = f'{scipy_openblas64.get_library()}.dll'
    shutil.copyfile(os.path.join(cblas_library_dir, dll_file),
                    os.path.join('kalman_filter', dll_file))
elif sys.platform == "darwin":
    import glob
    import shutil
    lib_dir = scipy_openblas64.get_lib_dir()
    dylibs = glob.glob(os.path.join(lib_dir, '*.dylib'))
    dylibs = glob.glob(os.path.join(lib_dir, '*.dll'))
    os.makedirs('kalman_filter/.dylibs', exist_ok=True)
    for lib in dylibs:
        shutil.copy(lib, 'kalman_filter/.dylibs')


# numpy c-api:
numpy_include_dir = np.get_include()

c_sources = [os.path.join(src_directory, x)
             for x in os.listdir(src_directory) if x.endswith('.c')]

module = Extension("kalman_filter.c_core", sources=c_sources,
                   libraries=libraries,
                   include_dirs=[numpy_include_dir,
                                 cblas_include_dir, src_directory, "."],
                   define_macros=[
                       ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                   library_dirs=[cblas_library_dir],
                   extra_link_args=extra_link_args,)

setup(
    ext_modules=[module],
)
