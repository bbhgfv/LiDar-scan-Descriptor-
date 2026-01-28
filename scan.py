from pathlib import Path
import numpy as np

from python.python.read_vel_sync import load_nclt_vel_sync
from loaders.descriptor import polar_hist_descriptor, load_descriptor_folder


def cosine_similarity_db(query_desc, db_descs):
    """
    query_desc: (D,)
    db_descs: (N, D)
    returns: (N,) similarities (higher = more similar)
    """
    return db_descs @ query_desc


def process_bin_to_descriptor(
    bin_file: Path,
    out_dir: Path,
    Nr: int = 30,
    Nphi: int = 60,
    r_max: float = 80.0,
    use_max_height: bool = True,
) -> tuple[np.ndarray, Path]:
    """
    Load a single NCLT velodyne_sync .bin file, compute a polar histogram descriptor,
    and save it as .npy into out_dir.

    Returns:
        desc: np.ndarray of shape (Nr*Nphi,)
        saved_path: Path to the saved .npy file
    """
    bin_file = Path(bin_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bin_file.exists():
        raise FileNotFoundError(f"File not found: {bin_file}")

    # 1) Load .bin -> (N,5) = x,y,z,intensity,ring
    pts = load_nclt_vel_sync(bin_file)
    xyz = pts[:, 0:3]  # (N,3)

    # 2) Compute descriptor -> (Nr*Nphi,)
    desc = polar_hist_descriptor(
        xyz,
        r_max=r_max,
        Nr=Nr,
        Nphi=Nphi,
        use_max_height=use_max_height,
    )

    # 3) Save descriptor
    saved_path = out_dir / f"{bin_file.stem}_Nr{Nr}_Nphi{Nphi}.npy"
    np.save(saved_path, desc)

    return desc, saved_path


if __name__ == "__main__":
    ROOT = Path(__file__).parent
    DATASET_DIR = ROOT / "datasets" / "2013-01-10" / "velodyne_sync"
    OUT_DIR = ROOT / "descriptor_results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bin_file = DATASET_DIR / "1357847240132236.bin"

    # get first num_of_files .bin files (sorted by timestamp)
    NUM_OF_FILES = 50
    bin_files = sorted(DATASET_DIR.glob("*.bin"))[:NUM_OF_FILES]

    print(f"Processing {len(bin_files)} LiDAR scans...")

    for i, bin_file in enumerate(bin_files, start=1):
        try:
            desc, saved_path = process_bin_to_descriptor(
                bin_file=bin_file,
                out_dir=OUT_DIR,
                Nr=30,
                Nphi=60,
                r_max=80.0,
                use_max_height=True,
            )

            print(
                f"[{i:02d}/{len(bin_files)}] "
                f"{bin_file.name} -> saved {saved_path.name} "
                f"(shape={desc.shape})"
            )

        except Exception as e:
            print(f"[{i:02d}/{len(bin_files)}] ERROR processing {bin_file.name}: {e}")
    
    # load all descriptors in OUT_DIR

    names, db_descs = load_descriptor_folder(OUT_DIR)
    print("Loaded DB:", db_descs.shape)   # (50, D)

    # -----------------------------
    # Matching & Retrieval

    query_idx = 0
    query_desc = db_descs[query_idx]
    query_name = names[query_idx]

    # cosine similarity
    sims = cosine_similarity_db(query_desc, db_descs)

    # remove self-match
    mask = np.arange(len(names)) != query_idx
    filtered_sims = sims[mask]
    filtered_names = [n for i, n in enumerate(names) if i != query_idx]

    # Top-K retrieval
    TOP_K = 5
    top_idx = np.argsort(-filtered_sims)[:TOP_K]

    print("\nQuery:", query_name)
    print("Top matches (cosine similarity):")

    for rank, i in enumerate(top_idx, start=1):
        print(f"{rank:02d}. {filtered_names[i]}   similarity={filtered_sims[i]:.4f}")



    print("Done.")
