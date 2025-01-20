from setuptools import setup, find_packages

setup(
    name="cached-binance-data",
    version="0.1.0",
    packages=find_packages(include=['cached_binance_data', 'cached_binance_data.*']),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "requests>=2.26.0",
        "python-dateutil>=2.8.2",
    ],
    extras_require={
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=2.0.0',
        ],
    },
    author="Kunthet",
    author_email="dev@kunthet.com",
    description="A Python module for downloading Binance futures market data with caching support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kunthet/cached-binance-data",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 