from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

# requirements = ["ipython>=6", "nbformat>=4", "nbconvert>=5", "requests>=2"]
requirements = []

setup(
    name="moopy",
    version="0.0.1", # Update __init__.py if the version changes!
    author="Derk Kappelle",
    author_email="derk_kappelle@live.nl",
    description="A package for Multi-Objective Optimization tools",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/moo4all/MooPy.git",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)