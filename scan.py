import numpy as np
import open3d as o3d
from pathlib import Path
from python.python.read_vel_sync import load_nclt_vel_sync
from loaders.descriptor import polar_hist_descriptor
import matplotlib.pyplot as plt
# -----------------------------
# Define paths

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "datasets" / "2013-01-10" / "velodyne_sync"

bin_file = DATASET_DIR / "1357847240132236.bin"
assert bin_file.exists(), f"File not found: {bin_file}"


pts = load_nclt_vel_sync(bin_file)      # (N,5)
xyz = pts[:, 0:3]                       # (N,3)
intensity = pts[:, 3].astype(np.uint8)  # (N,)
ring = pts[:, 4].astype(np.uint8)       # (N,)
points = xyz  # (N,3)
print(xyz.shape, intensity.min(), intensity.max(), ring.min(), ring.max())


# (A) show point cloud

#pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(points)
#o3d.visualization.draw_geometries([pcd])

# (B) compute descriptor
Nr, Nphi = 20, 60
desc = polar_hist_descriptor(xyz, r_max=80.0, Nr=Nr, Nphi=Nphi, use_max_height=True)
print("Descriptor shape:", desc.shape)

# (C) visualize descriptor
H = desc.reshape(Nr, Nphi)
plt.imshow(H, aspect="auto", origin="lower")
plt.xlabel("Angle bins")
plt.ylabel("Radius bins")
plt.colorbar(label="Max height (m)")
plt.title("Polar Histogram Descriptor")
plt.show()