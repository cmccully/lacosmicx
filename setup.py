'''
Created on May 26, 2011

@author: cmccully
'''
from distutils.core import setup, Extension

module1 = Extension('_lacosmicx',
                    
                    include_dirs = ['/Users/cmccully/cfitsio','/usr/stsci/pyssgx/Python2.7.1/include/python2.7','/usr/stsci/pyssgx/2.7.1/numpy/core/include/numpy','/usr/include/malloc'],
                    libraries = ['cfitsio','gomp'],
                    library_dirs = ['/Users/cmccully/cfitsio/lib'],
                    extra_compile_args=['-O3','-fopenmp','-funroll-loops'],
                    sources = ['lacosmicx.cpp','lacosmicx_py.cpp','functions.cpp']
                    )

setup (name = 'Lacosmicx',
       version = '1.0',
       author = 'Curtis McCully',
       author_email = 'cmccully@physics.rutgers.edu',
       ext_modules = [module1],
       py_modules=['lacosmicx'])
