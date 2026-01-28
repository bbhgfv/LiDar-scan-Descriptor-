import numpy as np

def polar_hist_descriptor(points_xyz, r_max=80.0, Nr=20, Nphi=60, use_max_height=True):
    """
    Simple global LiDAR descriptor:
    - bin points by (radius, angle) around the sensor
    - store either point count or max height per bin
    Returns: (Nr*Nphi,) normalized vector
    """
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]

    r = np.sqrt(x*x + y*y)
    phi = np.arctan2(y, x)  # [-pi, pi]

    # filter
    m = (r > 0.5) & (r < r_max) & (np.isfinite(z))
    r, phi, z = r[m], phi[m], z[m]

    # bins
    r_bin = np.floor(r / r_max * Nr).astype(int)
    r_bin = np.clip(r_bin, 0, Nr - 1)

    phi_norm = (phi + np.pi) / (2*np.pi)  # [0,1)
    phi_bin = np.floor(phi_norm * Nphi).astype(int)
    phi_bin = np.clip(phi_bin, 0, Nphi - 1)

    H = np.zeros((Nr, Nphi), dtype=np.float32)

    if use_max_height:
        # max height per bin
        for rb, pb, zz in zip(r_bin, phi_bin, z):
            if zz > H[rb, pb]:
                H[rb, pb] = zz
    else:
        # count points per bin
        np.add.at(H, (r_bin, phi_bin), 1.0)

    v = H.reshape(-1)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v
