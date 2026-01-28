def top_k_cosine(query_desc, db_descs, db_names, k=5):
    """
    query_desc: (D,)
    db_descs: (M, D)
    returns list of (name, score) sorted best->worst
    """
    q = query_desc.astype(np.float32).reshape(-1)
    # dot product = cosine similarity because vectors are normalized
    sims = db_descs @ q  # (M,)
    idx = np.argsort(-sims)[:k]
    return [(db_names[i], float(sims[i])) for i in idx]
