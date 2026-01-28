# LiDar-scan-Descriptor-

This project implements a simple LiDAR place-recognition pipeline using polar histogram descriptors and cosine similarity.

# How to Run

Place the NCLT LiDAR scans in:

datasets/2013-01-10/velodyne_sync/


Run the main script:

python scan.py

# What It Does

Loads Velodyne .bin LiDAR scans

Computes polar histogram descriptors (Nr × Nphi)

Saves descriptors to descriptor_results/

Builds a descriptor database

Retrieves the top-K most similar scans using cosine similarity


Descriptor images are for visualization only

Matching is performed on numeric descriptor vectors (.npy)
