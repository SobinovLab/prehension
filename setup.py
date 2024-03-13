#!python3.12
# -*- coding: utf-8 -*-
"""
Prehension Toolbox
Copyright 2024 Caleb A Raman, Anton Sobinov
https://github.com/BensmaiaLab/prehension_analysis
"""

from setuptools import setup, find_packages

setup(
    name='prehension',
    version='0.0.1',
    description='Prehension tools',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/BensmaiaLab/prehension_analysis',
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
    author='Caleb A Raman, Anton R Sobinov',
    author_email='craman@uchicago.edu',
    license='MIT',
    packages=find_packages())