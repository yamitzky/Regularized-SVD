from distutils.core import setup, Extension
from Cython.Build import cythonize
import Cython.Compiler.Options

import numpy as np

Cython.Compiler.Options.annotate = True

setup(
    name="Simon's Regularized SVD",
    include_dirs=[np.get_include()],
    ext_modules=cythonize(Extension(
        "rsvd",
        sources=["rsvd.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11", "-stdlib=libc++", "-mmacosx-version-min=10.7"],
        extra_link_args=["-std=c++11", "-stdlib=libc++", "-mmacosx-version-min=10.7"]
    ))
)
