import os
from distutils.core import setup
from setuptools import find_packages

cwd = os.getcwd()

setup(
    name="ecg-synth",
    version="1.0",
    description=(
        "This package contains the code for the project of forecasting ECG"
    ),
    author="Jeries Saleh",
    author_email="jeries.saleh@campus.technion.ac.il",
    url="",
    license="",
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Private",
        "Topic :: Software Development :: ECG",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    package_dir={"src": os.path.join(cwd, "src")},
    packages=find_packages(exclude=["baselines", "ptb_xl"]),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "matplotlib",
        "torch",
        "torchvision",
        "torchmetrics",
        "tensorboard",
        "tqdm",
        "h5py",
        "pandas",
        "pytest",
        "pytest-cov",
        "seaborn",
        "dicognito",
        "pydicom",
        "opt_einsum",
        "wfdb",
        "zuko",
        "torchdyn",
    ],
)
