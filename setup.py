import os
import sys
import subprocess

from setuptools.command.test import test as TestCommand

from setuptools import find_packages
from distutils.core import Extension
from distutils import log
from distutils.core import setup
from Cython.Build import cythonize

import numpy

CODELINES = """
import sys
from distutils.ccompiler import new_compiler
ccompiler = new_compiler()
ccompiler.add_library('gomp')
sys.exit(int(ccompiler.has_function('omp_get_num_threads')))
"""


def check_openmp():
    s = subprocess.Popen([sys.executable], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    stdout, stderr = s.communicate(CODELINES.encode('utf-8'))
    s.wait()
    return bool(s.returncode), (stdout, stderr)


def get_extensions():

    sources = ["lacosmicx/_lacosmicx.pyx", "lacosmicx/laxutils.c"]

    include_dirs = [numpy.get_include(), '.']

    libraries = []

    ext = Extension(name="_lacosmicx",
                    sources=sources,
                    include_dirs=include_dirs,
                    libraries=libraries,
                    language="c",
                    extra_compile_args=['-g', '-O3',
                                        '-funroll-loops', '-ffast-math'])

    has_openmp, outputs = check_openmp()
    if has_openmp:
        ext.extra_compile_args.append('-fopenmp')
        ext.extra_link_args = ['-g', '-fopenmp']
    else:
        log.warn('OpenMP was not found. '
                 'lacosmicx will be compiled without OpenMP. '
                 '(Use the "-v" option of setup.py for more details.)')
        log.debug(('(Start of OpenMP info)\n'
                   'compiler stdout:\n{0}\n'
                   'compiler stderr:\n{1}\n'
                   '(End of OpenMP info)').format(*outputs))

    return [ext]


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(name="lacosmicx",
      packages = find_packages(),
      ext_modules=cythonize(get_extensions()),
      tests_require=['pytest'],
      cmdclass={'test': PyTest})
