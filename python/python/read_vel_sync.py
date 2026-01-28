# !/usr/bin/python
#
# Example code to read a velodyne_sync/[utime].bin file
# Plots the point cloud using matplotlib. Also converts
# to a CSV if desired.
#
# To call:
#
#   python read_vel_sync.py velodyne.bin [out.csv]
#

import sys
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def convert(x_s, y_s, z_s, scaling=0.005, offset=-100.0):
    """
    Convert raw uint16 coordinates to meters (NCLT velodyne_sync format).
    """
    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset
    return x, y, z


def load_nclt_vel_sync(bin_path, scaling=0.005, offset=-100.0):
    """
    Load an NCLT velodyne_sync/[utime].bin file.

    File layout per point (8 bytes total):
      - x: uint16 (little endian)
      - y: uint16 (little endian)
      - z: uint16 (little endian)
      - intensity: uint8
      - laser_id: uint8

    Returns:
      points: np.ndarray of shape (N, 5) with columns:
              [x, y, z, intensity, laser_id] (x,y,z in meters)
    """
    points = []

    with open(bin_path, "rb") as f:
        while True:
            # Read one record (8 bytes)
            rec = f.read(8)
            if len(rec) < 8:
                break

            x_raw, y_raw, z_raw, intensity, laser_id = struct.unpack("<HHHBB", rec)
            x, y, z = convert(x_raw, y_raw, z_raw, scaling=scaling, offset=offset)

            points.append([x, y, z, intensity, laser_id])

    return np.asarray(points, dtype=np.float32)


def save_csv(points, out_csv_path):
    """
    Save points (Nx5) to CSV: x,y,z,intensity,laser_id
    """
    header = "x,y,z,intensity,laser_id"
    np.savetxt(out_csv_path, points, delimiter=",", header=header, comments="")


def plot_points(points, color_by="z"):
    """
    Quick 3D scatter plot.
    color_by: "z" (default) or None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]

    if color_by == "z":
        c = -zs
    else:
        c = None

    ax.scatter(xs, ys, -zs, c=c, s=5, linewidths=0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z (flipped)")
    plt.show()

import numpy as np

def main(argv):
    if len(argv) < 2:
        print("Usage: python read_vel_sync.py velodyne.bin [out.csv]")
        return 1

    bin_path = argv[1]
    out_csv = argv[2] if len(argv) >= 3 else None

    pts = load_nclt_vel_sync(bin_path)

    if out_csv:
        print("Writing to", out_csv)
        save_csv(pts, out_csv)

    plot_points(pts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))