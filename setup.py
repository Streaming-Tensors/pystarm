from setuptools import setup, Extension
import numpy as np
import pybind11

ext_modules = [
    Extension(
        'pystarm',
        ['starm.cpp'],
        include_dirs=[np.get_include(), pybind11.get_include()],
        language='c++'
    ),
]

setup(
    name='pystarm',
    ext_modules=ext_modules,
    zip_safe=False
)
