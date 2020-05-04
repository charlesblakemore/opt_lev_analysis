from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
	long_description = fh.read()

setup(name="opt_lev_analysis", version=2.0, 
      package_dir={"": "lib"},
      packages=find_packages(), 
      author="Charles Blakemore, Alexander Rider", 
      author_email="chas.blakemore@gmail.com",
      description="Analysis and Simulation for the Optical Levitation Project at Stanford",
      long_description=long_description,
      url="https://github.com/charlesblakemore/opt_lev_analysis")

