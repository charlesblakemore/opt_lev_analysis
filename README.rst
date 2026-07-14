
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

To install system-wide, noting the path to the src since no wheels
exist on PyPI, use::

   pip install ./opt_lev_analysis

If you intend to edit the code and want the import calls to reflect
those changes, install in developer mode::

   pip install -e opt_lev_analysis

If you don't want a global installation (i.e. if multiple users will
engage with and/or edit this library) and you don't want to use venv
or some equivalent::

   pip install -e opt_lev_analysis --user

where pip is pip3 for Python3 (tested on Python 3.6.9). Be careful 
NOT to use ``sudo``, as the latter two installations make a file
``easy-install.pth`` in either the global or the local directory
``lib/python3.X/site-packages/easy-install.pth``, and sudo will
mess up the permissions of this file such that uninstalling is very
complicated.


Uninstall
---------

If installed without ``sudo`` as instructed, uninstalling should be 
as easy as::

   pip uninstall opt_lev_analysis

If installed using ``sudo`` and with the ``-e`` and ``--user`` flags, 
the above uninstall will encounter an error.

Navigate to the file ``lib/python3.X/site-packages/easy-install.pth``, 
located either at  ``/usr/local/`` or ``~/.local`` and ensure there
is no entry for ``opt_lev_analysis``.


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