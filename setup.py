from setuptools import setup, find_packages

setup(
    name="black_scholes",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "streamlit>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    python_requires=">=3.8",
) 