
Analysis and Simulation for the Optical Levitation Project
==========================================================

This is a collection of analysis and simulation scripts developed to
facilitate the Optical Levitation Project at Stanford University, 
under the direction of Professor Giorgio Gratta. Although some of the
code is generally applicable to different analysis tasks, much of it
depends heavily on data files acquired for the project, and the 
content and structure inherent to those data files.

Install
-------

From sources
````````````

To install system-wide (needs administrator privilages) use::

   python setup.py install

If you intend to edit the code and want the import calls to reflect
those changes, install in developer mode::

   python setup.py develop

where python is a Python3 executable (tested on Python 3.6.9).

License
-------

The package is distributed under an open license (see LICENSE file for
information).

Related packages
----------------

`opt_lev_controls <https://github.com/stanfordbeads/opt_lev_controls>`_ - A companion 
library used on the data acquision computers associated to the project.

Authors
-------

Charles Blakemore (chas.blakemore@gmail.com),
Alexander Rider,
David Moore