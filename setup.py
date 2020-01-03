#!/usr/bin/env python

from distutils.core import setup

setup(
    name='fftracer',
    version='0.1.0',
    author='Josh Gardner',
    author_email='jpgard@cs.washington.edu',
    packages=['fftracer', 'fftracer.datasets', 'fftracer.training',
              'fftracer.utils'],
    url='https://github.com/jpgard/ffn-tracer',
    license='LICENSE',
    description='Flood-Filling Networks for 2D instance segmentation',
    long_description=open('README.md').read(),
    install_requires=['scikit-image', 'scipy', 'numpy', 'tensorflow', 'h5py', 'PIL',
                      'absl-py'],
)
