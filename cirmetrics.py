#!/usr/bin/env python3
"""Official CIRR / FashionIQ retrieval metrics.

Replicates /data/luozj/ACHN-Bicot/src/precompute_evaluation.py exactly:

  CIRR       R@1/5/10/50 with the reference image removed from the ranking,
             plus R_subset@1/2/3 over the query's ``img_set.members``.
  FashionIQ  R@1/5/10/50 per category; the official ``acc`` is the mean of the
             three categories' (R@10 + R@50) / 2.

Inputs are a full similarity matrix (n_queries x n_corpus); ranking is
descending similarity.  Keeping the matrix (instead of pre-sorted indices) makes
the CIRR reference-exclusion trivial.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _rank_order(sim: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Corpus indices in descending-similarity order, one row per query.

    The matrix is copied first so that masking (dropping the reference image)
    never mutates the caller's scores.  Its dtype is left as-is: the scores are
    float32 and sorting their lossless float64 upcast ranks them identically.
    """
    sim = np.array(sim, copy=True)
    if mask is not None:
        sim[mask] = -np.inf
    return np.argsort(-sim, axis=1)


def _recalls_idx(order: np.ndarray, target_idx: np.ndarray,
                 ks: Sequence[int]) -> Dict[str, float]:
    out = {}
    for k in ks:
        hit = (order[:, :k] == target_idx[:, None]).any(axis=1)
        out["R@%d" % k] = float(hit.mean() * 100.0)
    return out


def cirr_metrics(sim: np.ndarray, corpus_ids: Sequence[str],
                 ref_ids: Sequence[str], target_ids: Sequence[str],
                 subsets: Sequence[Sequence[str]]) -> Dict[str, float]:
    """CIRR R@k + R_subset@k.  Exactly one target per query."""
    name_to_idx = {n: i for i, n in enumerate(corpus_ids)}
    rows = np.arange(np.asarray(sim).shape[0])
    mask = None
    if len(ref_ids):
        mask = (rows, [name_to_idx[r] for r in ref_ids])
    order = _rank_order(sim, mask)
    tgt = np.array([name_to_idx[t] for t in target_ids], dtype=np.int64)
    res = _recalls_idx(order, tgt, (1, 5, 10, 50))

    # R_subset@k: restrict each query's ranking to its own img_set.members
    n = order.shape[0]
    group_hits = {1: np.zeros(n, dtype=bool), 2: np.zeros(n, dtype=bool),
                  3: np.zeros(n, dtype=bool)}
    for i, members in enumerate(subsets):
        midx = np.array([name_to_idx[m] for m in members if m in name_to_idx],
                        dtype=np.int64)
        kept = order[i][np.isin(order[i], midx)]  # already in rank order
        for k in (1, 2, 3):
            group_hits[k][i] = bool((kept[:k] == tgt[i]).any())
    for k in (1, 2, 3):
        res["R_subset@%d" % k] = float(group_hits[k].mean() * 100.0)

    res["acc"] = (res["R_subset@1"] + res["R@5"]) / 2.0
    res["mean"] = float(np.mean([res["R@1"], res["R@5"], res["R@10"], res["R@50"],
                                 res["R_subset@1"], res["R_subset@2"],
                                 res["R_subset@3"]]))
    return res


def fiq_metrics(sim: np.ndarray, corpus_ids: Sequence[str],
                target_ids: Sequence[str]) -> Dict[str, float]:
    """FashionIQ R@k for ONE category; exactly one target per query."""
    name_to_idx = {n: i for i, n in enumerate(corpus_ids)}
    order = _rank_order(sim)
    tgt = np.array([name_to_idx[t] for t in target_ids], dtype=np.int64)
    res = _recalls_idx(order, tgt, (1, 5, 10, 50))
    res["acc"] = (res["R@10"] + res["R@50"]) / 2.0
    return res


def aggregate_fiq(per_category: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Official FashionIQ score = plain mean over dress / shirt / toptee."""
    cats = list(per_category)
    out = {"acc": float(np.mean([per_category[c]["acc"] for c in cats]))}
    for k in ("R@1", "R@5", "R@10", "R@50"):
        out[k] = float(np.mean([per_category[c][k] for c in cats]))
    return out
