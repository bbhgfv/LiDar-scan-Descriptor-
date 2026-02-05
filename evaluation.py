import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def load_gt_xy(gt_csv: Path) -> pd.DataFrame:
    """
    Returns DataFrame with columns: ts, x, y
    Handles 'no header' GT CSV like your screenshot.
    """
    df = pd.read_csv(gt_csv, header=None)
    # col0=timestamp, col1=x, col2=y (based on your screenshot)
    df = df.rename(columns={0: "ts", 1: "x", 2: "y"})
    df = df[["ts", "x", "y"]].copy()
    df["ts"] = df["ts"].astype(np.int64)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def ts_from_desc_path(p: Path) -> int:
    # takes the first chunk before '_' as timestamp
    return int(p.stem.split("_")[0])


def build_nearest_gt_lookup(gt_df: pd.DataFrame):
    """
    Prepares arrays for fast nearest neighbor lookup by timestamp.
    """
    ts = gt_df["ts"].to_numpy(np.int64)
    xy = gt_df[["x", "y"]].to_numpy(np.float64)
    return ts, xy

def nearest_xy(ts_query: int, gt_ts: np.ndarray, gt_xy: np.ndarray, max_dt: int = 50_000) -> np.ndarray | None:
    """
    Find nearest ground-truth xy for ts_query using binary search.
    max_dt is in microseconds (NCLT timestamps are usually in µs).
    If nearest is too far in time, return None.
    """
    i = np.searchsorted(gt_ts, ts_query)
    candidates = []
    if 0 <= i < len(gt_ts):
        candidates.append(i)
    if 0 <= i-1 < len(gt_ts):
        candidates.append(i-1)

    if not candidates:
        return None

    best_i = min(candidates, key=lambda j: abs(int(gt_ts[j]) - int(ts_query)))
    if abs(int(gt_ts[best_i]) - int(ts_query)) > max_dt:
        return None
    return gt_xy[best_i]


def load_desc_db(desc_dir: Path, num_files: int = 50):
    
    paths = sorted(desc_dir.glob("*.npy"))[:num_files]   # you said 50 results
    descs = np.stack([np.load(p) for p in paths], axis=0).astype(np.float32)
    ts = np.array([ts_from_desc_path(p) for p in paths], dtype=np.int64)
    return paths, ts, descs

def topk_cosine(query_desc: np.ndarray, db_descs: np.ndarray, k: int = 5, exclude_index: int | None = None):
    sims = db_descs @ query_desc  # (N,)
    if exclude_index is not None:
        sims[exclude_index] = -np.inf
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


def recall_at_k_with_radius(
    db_descs: np.ndarray,
    db_ts: np.ndarray,
    gt_ts: np.ndarray,
    gt_xy: np.ndarray,
    K: int = 5,
    radius_m: float = 10.0,
    max_dt_us: int = 50_000,
) -> float:
    """
    For each item as query: retrieve top-K by descriptor similarity.
    Count success if ANY of top-K is within radius_m in GT position space.
    """

    exclude_dt_us = 30_000_000  # 30 seconds in microseconds (example)
    N = db_descs.shape[0]
    successes = 0
    valid_queries = 0

    for i in range(N):
        q_ts = int(db_ts[i])
        q_xy = nearest_xy(q_ts, gt_ts, gt_xy, max_dt=max_dt_us)
        if q_xy is None:
            continue

        valid_queries += 1
        idx, _ = topk_cosine(db_descs[i], db_descs, k=K, exclude_index=i)

        

        hit = False
        for j in idx:
            if abs(int(db_ts[j]) - q_ts) < exclude_dt_us:
                continue
            cand_ts = int(db_ts[j])
            cand_xy = nearest_xy(cand_ts, gt_ts, gt_xy, max_dt=max_dt_us)
            if cand_xy is None:
                continue
            if np.linalg.norm(q_xy - cand_xy) <= radius_m:
                hit = True
                break

        if hit:
            successes += 1

    if valid_queries == 0:
        return 0.0
    return successes / valid_queries


def visualize_eval_examples(
    db_descs: np.ndarray,
    db_ts: np.ndarray,
    gt_ts: np.ndarray,
    gt_xy: np.ndarray,
    K: int = 5,
    radius_m: float = 10.0,
    max_dt_us: int = 50_000,
    n_examples: int = 5,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    # pick queries that actually have GT
    valid_idx = []
    for i in range(len(db_ts)):
        q_xy = nearest_xy(int(db_ts[i]), gt_ts, gt_xy, max_dt=max_dt_us)
        if q_xy is not None:
            valid_idx.append(i)

    if len(valid_idx) == 0:
        print("No valid queries (no GT matches found). Increase max_dt_us or check GT file.")
        return

    chosen = rng.choice(valid_idx, size=min(n_examples, len(valid_idx)), replace=False)

    # plot background: all GT positions of your 50 scans
    all_xy = []
    for t in db_ts:
        p = nearest_xy(int(t), gt_ts, gt_xy, max_dt=max_dt_us)
        if p is not None:
            all_xy.append(p)
    all_xy = np.array(all_xy)

    plt.figure(figsize=(8, 8))
    if len(all_xy) > 0:
        plt.scatter(all_xy[:, 0], all_xy[:, 1], s=8, alpha=0.3)
        plt.title(f"Evaluation visualization (Top-{K}, R={radius_m}m)")

    for qi in chosen:
        q_desc = db_descs[qi]
        q_ts = int(db_ts[qi])
        q_xy = nearest_xy(q_ts, gt_ts, gt_xy, max_dt=max_dt_us)
        if q_xy is None:
            continue

        # retrieve top-K (cosine), exclude itself
        idx, sims = topk_cosine(q_desc, db_descs, k=K, exclude_index=qi)

        # plot query
        plt.scatter(q_xy[0], q_xy[1], s=120, marker="x")
        plt.text(q_xy[0], q_xy[1], f"Q\n{q_ts}", fontsize=8)

        # plot matches
        for rank, j in enumerate(idx, start=1):
            m_ts = int(db_ts[j])
            m_xy = nearest_xy(m_ts, gt_ts, gt_xy, max_dt=max_dt_us)
            if m_xy is None:
                continue

            d = float(np.linalg.norm(q_xy - m_xy))
            ok = d <= radius_m

            # line from query to match
            plt.plot([q_xy[0], m_xy[0]], [q_xy[1], m_xy[1]], linewidth=1)

            # mark match point
            plt.scatter(m_xy[0], m_xy[1], s=60, marker="o")

            # annotate
            plt.text(
                m_xy[0], m_xy[1],
                f"{rank}:{d:.1f}m",
                fontsize=7
            )

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.show()



if __name__ == "__main__":
    ROOT = Path(__file__).parent
    DESC_DIR = ROOT / "descriptor_results"
    GT_CSV   = ROOT / "groundtruth_2013-01-10.csv"   # change path if needed

    gt_df = load_gt_xy(GT_CSV)
    gt_ts, gt_xy = build_nearest_gt_lookup(gt_df)

    paths, db_ts, db_descs = load_desc_db(DESC_DIR, 1000)

    for K in [1, 5, 10]:
        r = recall_at_k_with_radius(
            db_descs=db_descs,
            db_ts=db_ts,
            gt_ts=gt_ts,
            gt_xy=gt_xy,
            K=K,
            radius_m=10.0,
            max_dt_us=50_000,   # adjust if needed
        )
        print(f"Recall@{K} (R<=10m): {r:.3f}")

     # visualization for intuition
    visualize_eval_examples(
        db_descs=db_descs,
        db_ts=db_ts,
        gt_ts=gt_ts,
        gt_xy=gt_xy,
        K=5,
        radius_m=10.0,
        max_dt_us=50_000,
        n_examples=5,
        seed=0,
    )
