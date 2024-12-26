from setuptools import setup, find_packages

setup(
    name="fracstack",
    version="0.1.0",
    author="DillyDilly",
    author_email="aidend@uoregon.edu",
    description="A package for fractal analysis and box counting with visualization capabilities",
    packages=find_packages(),
    url="https://github.com/apdill/FracStack.git",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scikit-image>=0.18.0",
        "scipy>=1.6.0",
        "tqdm>=4.50.0",
        "scikit-learn>=0.24.0",
        "pandas>=2.2.0"
    ],
)