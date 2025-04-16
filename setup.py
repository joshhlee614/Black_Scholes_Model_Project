from setuptools import setup, find_packages

setup(
    name="black_scholes",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.0",
        "scipy==1.10.0",
        "streamlit==1.24.0",
        "pandas==2.0.0",
        "matplotlib==3.7.0",
        "seaborn==0.12.0",
        "plotly==5.13.0",
        "sqlalchemy==2.0.0",  # Required for database operations
    ],
    extras_require={
        "dev": [
            "pytest==7.3.1",
            "black==23.3.0",
            "flake8==6.0.0",
        ],
    },
    python_requires=">=3.8",
    author="Joshua Lee",
    author_email="joshhlee614@gmail.com",
    description="A comprehensive implementation of the Black-Scholes option pricing model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/joshhlee614/Black_Scholes_Model_Project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords="black-scholes, options, pricing, finance, quantitative",
) 