from distutils.core import setup
from Cython.Build import cythonize

import numpy

setup(
    ext_modules=cythonize("c_graph.pyx"),
    include_dirs=[numpy.get_include()]
)
