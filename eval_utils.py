"""Small evaluation helpers used by ``calico_eval.py``.

Both functions are copied verbatim from the SDR-CIR codebase so that the
numbers reported by CALICO are computed by exactly the same code paths:

  * ``circo_metrics`` -- from ``sdrcir_ref/view_eval_cirbench.py`` (the official
    CIRCO mAP@k, itself verbatim ``compute_results.py::circo``).
  * ``simplex``       -- from ``sdrcir_ref/view_eval.py`` (the non-negative
    weight simplex on a ``step`` grid used by the multi-view fusion grid search).
"""
import numpy as np

# CIRCO reports mAP / R at these cut-offs.
RETRIEVALS = (5, 10, 25, 50)


def circo_metrics(sim, corpus_ids, ref_ids, target_sets, mask_ref=True):
    """Official CIRCO mAP@k: sum of precisions *at GT ranks* / min(|GT|, k).

    Verbatim ``compute_results.py::circo``: the cumulative precision is zeroed
    at every non-ground-truth rank (``* map_labels``) before summing.
    """
    sim = np.array(sim, copy=True)
    name_to_idx = {n: i for i, n in enumerate(corpus_ids)}
    rows = np.arange(sim.shape[0])
    if mask_ref:
        sim[rows, [name_to_idx[r] for r in ref_ids]] = -np.inf
    order = np.argsort(-sim, axis=1)
    # index-based ground-truth labels: identical to comparing corpus names, but
    # avoids materialising an (n, N) string array (which is ~GBs on CIRCO).
    gt_idx = [np.array([name_to_idx[t] for t in gts if t in name_to_idx],
                       dtype=np.int64) for gts in target_sets]
    res = {}
    acc = {k: 0.0 for k in RETRIEVALS}
    racc = {k: 0.0 for k in RETRIEVALS}
    for i, gts in enumerate(target_sets):
        g = set(gts)
        lab = np.isin(order[i], gt_idx[i]).astype(np.float32)
        prec = np.cumsum(lab) * lab / np.arange(1, len(lab) + 1)
        for k in RETRIEVALS:
            acc[k] += prec[:k].sum() / min(len(g), k)
            racc[k] += float(lab[:k].sum() > 0)
    n = len(target_sets)
    for k in RETRIEVALS:
        res["mAP@%d" % k] = acc[k] / n * 100.0
        res["R@%d" % k] = racc[k] / n * 100.0
    return res


def simplex(step, k):
    """all non-negative k-tuples on the grid that sum to 1"""
    g = int(round(1.0 / step))
    out = []

    def rec(pref, left, depth):
        if depth == k - 1:
            out.append(tuple([x * step for x in pref] + [left * step]))
            return
        for i in range(left + 1):
            rec(pref + [i], left - i, depth + 1)
    rec([], g, 0)
    return out
