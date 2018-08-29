# Etec
Files for the implementation of the E-tec algorithm (Ensemble-based Topological Entropy Calculation)

This repository includes two main files: Etec2Da.py and Etec2Db.py.  They are essentially the same, except that version "b" can impliment a constrained Delauney triangulation, (which requires the triangle module http://dzhelil.info/triangle/delaunay.html, unfortunately constrainded Delaunay is not a part of the standard scipy.spatial).  Version "a" will simply require that either the initial band you'd like to evolve forward has all edges in the Delaunay triangulation, or that you run it in the "mesh" mode, where every initial edge is a band segment.

Additional files include a jupyter/ipython notebook, which serves as an example for how to run E-tec.

The DOI for this (via Zenodo) is :https://zenodo.org/badge/latestdoi/146612307
