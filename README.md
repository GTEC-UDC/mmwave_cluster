# README

This repository includes several tools and ROS nodes to cluster mmWave cloud points.
The nodes included in the repository are:

* **optics_cluster_simple**:  This node uses OPTICS algorithm to perform the clustering.
* **dbscan_cluster**:  This node uses DBSCAN algorithm to perform the clustering.
* **hierarchy_cluster**:  This node uses HIERARCHY clustering algorithm to perform the clustering.
* **overlap_cluster**:  This node can fuse clusters with nearby centroids.

The ```launch``` folder contains some launch files to launch the clustering algorithms with some parameters.

This repository is related with the next paper. Please cite us if this code is useful to you.

Barral, V., Dominguez-Bolano, T., Escudero, C. J., & Garcia-Naya, J. A. *An IoT System for Smart Building Combining Multiple mmWave FMCW Radars Applied to People Counting.*