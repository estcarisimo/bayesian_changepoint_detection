#!/usr/bin/env python3
"""
Setup script for bayesian-changepoint-detection package.

This setup.py is maintained for backward compatibility.
The main configuration is now in pyproject.toml.
"""

from setuptools import setup, find_packages

# Read version from __init__.py
def get_version():
    """Get version from package __init__.py file."""
    with open('bayesian_changepoint_detection/__init__.py', 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '1.0.0'

# Read README for long description
def get_long_description():
    """Get long description from README.md file."""
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Bayesian changepoint detection algorithms with PyTorch support"

setup(
    name='bayesian-changepoint-detection',
    version=get_version(),
    description='Bayesian changepoint detection algorithms with PyTorch support',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Johannes Kulick',
    author_email='mail@johanneskulick.net',
    url='https://github.com/hildensia/bayesian_changepoint_detection',
    packages=find_packages(),
    python_requires='>=3.8.1',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'isort>=5.12.0',
            'flake8>=6.1.0',
            'mypy>=1.0.0',
            'jupyter>=1.0.0',
            'notebook>=6.4.0',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.2.0',
            'numpydoc>=1.5.0',
        ],
        'gpu': [
            'torch[cuda]>=2.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='bayesian changepoint detection time-series pytorch',
    project_urls={
        'Homepage': 'https://github.com/hildensia/bayesian_changepoint_detection',
        'Repository': 'https://github.com/hildensia/bayesian_changepoint_detection',
        'Issues': 'https://github.com/hildensia/bayesian_changepoint_detection/issues',
    },
)
