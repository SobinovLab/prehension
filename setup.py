#!python3.12
# -*- coding: utf-8 -*-
"""
Prehension Toolbox
Copyright 2024 Caleb A Raman, Anton Sobinov
https://github.com/BensmaiaLab/prehension
"""

from setuptools import setup, find_packages

setup(
    name='prehension',
    version='0.0.1',
    description='Tools for processing and extracting kinematic and kinetic data from prehension experiments',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/BensmaiaLab/prehension',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'moviepy',
        'opencv-contrib-python',
        'reportlab',
        'pyyaml',
        'easygui',
        'astropy'],
    author='Anton R Sobinov, Caleb A Raman, Charles M Greenspon',
    author_email='an.sobinov@gmail.com',
    license='MIT',
    packages=find_packages())
