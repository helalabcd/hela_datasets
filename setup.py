from setuptools import setup, find_packages

setup(
    name="Helalab datasets",
    version="0.1.0",
    author="Constantin Dalinghaus",
    author_email="dalinghaus.constantin@gmail.com",
    description="Various dataset classes for Hela tracking data",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/helalabcd/hela_datasets",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        # List your package dependencies here
        # "somepackage>=1.0",
    ],
)
