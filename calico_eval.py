#!/usr/bin/env python3
"""CALICO -- Cross-query Adaptive Late Interaction with COlumn calibration.

Training-free re-ranking for zero-shot composed image retrieval (ZS-CIR).  The
single-vector ZS-CIR score is left *untouched* as the base ranker; CALICO adds a
set of late-interaction channels on top and then calibrates the score matrix
with two label-free, cross-query column operators:

    S(q,g) = base(q,g)
             + sum_k  w_k * z(M_k(q,g)) * sd(base)          (variance matched)

    M_X(q,g) = (1/|top frac|) sum_{x in topK(frac)} max_{t in G_g} <x, t>

Channels, all cut from the M1 role structure and the R3 edit axis:
    rw   = the rewrite view (what the user asks for)
    de   = extra visual detail (colour / material / pattern / shape)
    O    = union of ``ops[].new``   -- what the target must contain
    Om   = union of ``ops[].old``   -- what the target must NOT still contain
    A    = ``preserve``             -- the textual twin of the visual anchor
    ctx  = ``scene``                -- background / context
    R3   = edit-direction consistency (cosdisp_xc): the edit axis
           d = normalize(t_new - t_old) against each candidate's image-space
           displacement, with the cross-query column mean removed.

Two applications of the column operator  x <- x - lam * col_stat(x)  (module 2):
    fine scale   -- CSLS: subtract lam*r_g(g), r_g(g) = mean of the k largest
                    column entries over the query rows (--csls k --csls_lam 0.5)
    coarse scale -- gallery-size-adaptive penalty on the raw fused score, with
                    k = clip(round(rho*N), 20, N//50) and a single rho for every
                    dataset (--r5_ms_rho).

Everything is fitted under the k-fold CV protocol (fold = arange(n) % folds,
held-out scoring), exactly as the auxiliary baselines, so the numbers are
directly comparable.  The frozen paper configuration is enabled by one flag
(--cqcc):

Usage
  python calico_eval.py --dataset cirr_val  --cqcc --csls 12 --csls_lam 0.5 \
      --gpu 0 --out out/cirr_val.json
  python calico_eval.py --dataset circo_test --cqcc --csls 20 --csls_lam 0.5 \
      --submit --hparams out/circo_val.json --sub-dir submit
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------- paths ------
# Everything is resolved relative to this folder so the package is portable.
# Set CALICO_ROOT to relocate the whole package, or use the per-path CLI flags
# (--manifest / --index-dir / --patch / --views / --roles / --desc) to point at
# data that lives elsewhere.
ROOT = Path(os.environ.get("CALICO_ROOT", Path(__file__).resolve().parent)).resolve()
HERE = ROOT
MAN_DIR = ROOT / "manifests"          # bundled cirbench manifests
ART_DIR = ROOT / "artifacts"          # MLLM text views / roles / descriptions
IDX_DIR = ROOT / "indices" / "cirbench"
PATCH_DIR = ROOT / "out" / "lateint"
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT))
LAION_W = MODELS / "clip_laion" / "open_clip_pytorch_model.bin"

# Backbone column for the SDR-CIR-style B/32, L/14, G/14 sweep.  L14 is the
# frozen project default: its index tag and patch-cache filename are unchanged,
# so omitting --clip reproduces every historical number bit-for-bit.
CLIP_SPECS = {
    "L14": dict(model="ViT-L-14",
                weights=MODELS / "clip_laion" / "open_clip_pytorch_model.bin",
                idx_tag="cliplaion", patch_suffix=""),
    "B32": dict(model="ViT-B-32",
                weights=MODELS / "clip_laion_b32" / "open_clip_pytorch_model.bin",
                idx_tag="cliplaion_b32", patch_suffix=".cliplaion_b32"),
    "G14": dict(model="ViT-bigG-14",
                weights=MODELS / "clip_laion_g14" / "open_clip_pytorch_model.bin",
                idx_tag="cliplaion_g14", patch_suffix=".cliplaion_g14"),
}

from cirmetrics import aggregate_fiq, cirr_metrics, fiq_metrics  # noqa: E402
from eval_utils import circo_metrics, simplex  # noqa: E402
from faiss_utils import load_faiss_index  # noqa: E402

# per-dataset defaults, copied verbatim from view_eval_cirbench.py
DATASETS = {
    "cirr_val": ("selectivecot_qwen3vl8b_cirr_val.json", (0.05, 0.50)),
    "cirr_test1": ("selectivecot_qwen3vl8b_cirr_test1.json", (0.05, 0.50)),
    "circo_val": ("circo_val_qwen3vl8b_desc_anchor.json", (0.15, 0.35)),
    # test split: same 123,403-image COCO unlabeled gallery, ground truth withheld
    # by the organisers, so this entry only ever feeds --submit.
    "circo_test": ("circo_test_qwen3vl8b_desc_anchor.json", (0.15, 0.35)),
    "fiq_dress_val": ("selectivecot_qwen3vl8b_fiq_dress_val.json", (0.20, 0.40)),
    "fiq_shirt_val": ("selectivecot_qwen3vl8b_fiq_shirt_val.json", (0.20, 0.40)),
    "fiq_toptee_val": ("selectivecot_qwen3vl8b_fiq_toptee_val.json", (0.20, 0.40)),
    "pinpoint": ("pinpoint_qwen3vl8b_desc.json", (0.05, 0.50)),
}
ABGRID = [(al, be) for al in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
          for be in (0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.65, 0.8, 0.95, 1.1,
                     1.3, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.4)]
G_REF = (0.0, 0.02, 0.05, 0.1, 0.2)
POOLS = {"mean": ("mean", 1.0), "top50": ("top", 0.50), "max": ("top", 1e-6)}
WGRID = (0.0, 0.05, 0.1, 0.2, 0.4, 0.75)
# weights for the R5' reference-anchored k-reciprocal channel: the Jaccard score
# already lives on a z-scored scale, so the grid spans "off" to "dominant".  The
# optimum sits below 0.1 and the curve falls off fast past it, so the low end is
# sampled finely (see r5kr_probe.py for the full weight curve).
KRGRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
# "Om" (what the target must no longer contain) is a *negative* evidence channel:
# a high MaxSim against it means the old attribute is still there, so it can only
# enter the score with a non-positive weight.
NGRID = {"Om": (0.0, -0.02, -0.05, -0.1, -0.2, -0.4)}


def norm_rows(x, eps=1e-8):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def zrow(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)


def orth_resid(y, xs, ridge=1e-6, bq=32):
    """Per-query least-squares residual of ``y`` against the span of ``xs``.

    Both ``y`` and every entry of ``xs`` are (n, N) score matrices over the *same*
    candidate columns, so the projection is fitted row by row: candidate g in query
    i keeps only the part of ``y[i, g]`` that the other channels cannot already
    predict.  This is the linear-algebra form of "give R3 the floor only for what
    the channels it is redundant with have not said".
    """
    y = np.asarray(y, np.float32)
    xs = [np.asarray(x, np.float32) for x in xs]
    if not xs:
        return y
    # A basis entry may be a gallery-only column statistic, i.e. shape (1, N); every
    # query projects against the same row, so broadcast it instead of dropping it.
    xs = [np.broadcast_to(x, y.shape) if x.shape[0] != y.shape[0] else x for x in xs]
    m = len(xs)
    eye = ridge * np.eye(m, dtype=np.float64)
    out = np.empty_like(y)
    for i in range(0, y.shape[0], bq):
        j = min(i + bq, y.shape[0])
        X = np.stack([x[i:j] for x in xs], axis=2).astype(np.float64)   # (b, N, m)
        yy = y[i:j].astype(np.float64)                                  # (b, N)
        XtX = np.einsum("nmk,nml->nkl", X, X)
        Xty = np.einsum("nmk,nm->nk", X, yy)
        b = np.linalg.solve(XtX + eye, Xty[..., None])[..., 0]
        out[i:j] = (yy - np.einsum("nmk,nk->nm", X, b)).astype(np.float32)
    return out


def r5dual_channels(baseS, C, ref_idx, n, N, dataset, k1, k2):
    """Dual-anchor k-reciprocal channels for the merged R5 module.

    Returns ``(kr_ref, kr_top1, medrank)``:
      * ``kr_ref``   -- Jaccard channel whose probe is the *reference image*;
      * ``kr_top1``  -- Jaccard channel whose probe is the base score's own top-1
                        (a query-expansion style anchor);
      * ``medrank``  -- ``median_rank(base top-1 inside kr_ref) / N``, a label-free
                        gallery-size consistency gauge used to gate ``kr_ref``.
    The gallery-side encoding is read from the cached kNN graph (no re-encode).
    """
    import r5_probe_lab as RPL
    ar = np.arange(n)
    Gk = RPL.load_cache(RPL.cache_path(dataset, k1, k2))
    kr_ref = np.stack([RPL.ref_channel(Gk, C, int(ref_idx[i]))
                       for i in range(n)]).astype(np.float32)
    kr_ref[ar, ref_idx] = 0.0
    b2 = np.array(baseS, dtype=np.float32, copy=True)
    b2[ar, ref_idx] = -np.inf
    top1 = np.argmax(b2, axis=1)
    pv = norm_rows(C[top1].astype(np.float32))
    kr_top1 = np.stack([RPL.probe_channel(pv[i], C, Gk, seed="thr")[0]
                        for i in range(n)]).astype(np.float32)
    kr_top1[ar, ref_idx] = 0.0
    rk = np.argsort(np.argsort(-kr_ref, axis=1), axis=1)
    medrank = float(np.median(rk[ar, top1])) / N
    return kr_ref, kr_top1, medrank


def r5dual_gate_value(kr_ref, medrank, N, mode, med_thr, nz_thr, min_n=0):
    """Label-free service-interval gauge for the kr_ref anchor.

    ``medrank`` : median_rank(base top-1 inside kr_ref) / N  -- small means the
                  reference's neighbourhood is consistent with the base ranking.
    ``nz``      : mean over queries of the fraction of gallery rows with
                  kr_ref > 0.01, i.e. how *broad* the reference neighbourhood is.
                  A broad set means "everything that looks like the reference",
                  so the channel cannot carry edit-specific evidence (CIRCO).
    Returns ``(gate, gauge)``; the gate is 1 when the split is inside the
    module's service interval.
    """
    if mode == "nz":
        nz = float((kr_ref > 0.01).mean())
        # The anchor is a *reference-conditioned* statistic: it is only trusted
        # when the gallery is large enough (N >= min_n) for a fixed k-reciprocal
        # neighbourhood to stay discriminative AND the reference support is broad
        # enough (nz >= nz_thr).  A tiny gallery (CIRR, N=2297) fails the first
        # test, a diffuse support (CIRCO) fails the second, so the block is
        # switched off there and only the multi-scale column term remains.
        ok = (nz >= nz_thr) and (N >= min_n)
        return (1.0 if ok else 0.0), nz
    return (1.0 if medrank <= med_thr else 0.0), float(medrank)


_STOP = set("""a an the of and or with without for in on at to from is are was were be
been being this that these those it its his her their there here as by into over under
than then so such very more most less least not no none same other others two three
one person people image photo picture shows show showing has have had having""".split())
_WORD = re.compile(r"[a-z0-9]+")


def _toks(s):
    return {w for w in _WORD.findall(str(s).lower()) if w not in _STOP and len(w) > 1}


def _ov(a, b):
    """Number of shared content words -- the grounding test of the role audit."""
    return len(_toks(a) & _toks(b))


def main():
    ap = argparse.ArgumentParser(
        description="CALICO - cross-query adaptive late interaction with column "
                    "calibration for composed image retrieval.")
    # ---- what to evaluate, and where the data lives -------------------------
    ap.add_argument("--dataset", default="cirr_val",
                    help="manifest name, e.g. cirr_val, cirr_test1, circo_test, fiq_shirt_val")
    ap.add_argument("--clip", default="L14", choices=sorted(CLIP_SPECS),
                    help="CLIP backbone column: L14=ViT-L/14 LAION (default, historical "
                         "config), B32, G14")
    ap.add_argument("--manifest", default=None,
                    help="cirbench manifest JSON (default: manifests/<dataset>.json)")
    ap.add_argument("--index-dir", default=None,
                    help="FAISS index dir (default: indices/cirbench/<dataset>/<idx_tag>)")
    ap.add_argument("--patch", default=None,
                    help="gallery patch-token cache npz (default: out/lateint/<dataset>_patch*.npz)")
    ap.add_argument("--views", default=None,
                    help="MLLM text-view JSON (default: artifacts/views/views_qwen3vl8b_<dataset>.json)")
    ap.add_argument("--roles", default=None,
                    help="MLLM role-view JSON (default: artifacts/roles/roles_qwen3vl8b_<dataset>.json)")
    ap.add_argument("--desc", default=None,
                    help="MLLM description JSON (default: artifacts/desc/<dataset default>)")

    # ---- the method ---------------------------------------------------------
    ap.add_argument("--cqcc", action="store_true",
                    help="enable the frozen CALICO config with ONE flag: fine-scale "
                         "cross-query CSLS column calibration (pair with --csls k "
                         "--csls_lam 0.5) + coarse gallery-scale-adaptive column "
                         "calibration (k = clip(round(0.018*N), 20, N//50), one rho for "
                         "every dataset) + the R3 edit-direction channel (cosdisp_xc). "
                         "The SAME method runs on every dataset; the only size-dependent "
                         "quantity is k.")
    ap.add_argument("--csls", type=int, default=0,
                    help="fine-scale cross-query CSLS calibration: subtract lam*r_g(g) "
                         "from the fused score, where r_g(g) is the mean of the k largest "
                         "column entries over the query rows (0 = off; the paper uses "
                         "k=12 on CIRR and k=20 on CIRCO / FashionIQ). Cross-query, "
                         "label-free.")
    ap.add_argument("--csls_lam", type=float, default=0.5)
    ap.add_argument("--r5_ms", type=float, default=0.0,
                    help="coarse-scale column-penalty weight (--cqcc sets 0.120)")
    ap.add_argument("--r5_ms_rho", type=float, default=0.0,
                    help="gallery-size rule for the coarse scale: k = clip(round(rho*N), "
                         "20, N//50); one rho is valid on every dataset (--cqcc sets 0.018)")
    ap.add_argument("--r3_mode", default="off",
                    help="R3 edit-direction channel (default off; --cqcc enables cosdisp_xc, "
                         "the CALICO direction-axis instance of the column operator)")

    # ---- runtime / output ---------------------------------------------------
    ap.add_argument("--tokens", type=int, default=256, help="gallery patch tokens per image")
    ap.add_argument("--gbs", type=int, default=1024, help="gallery images streamed per chunk")
    ap.add_argument("--sub", type=int, default=128, help="images per MaxSim block")
    ap.add_argument("--qbs", type=int, default=32, help="queries per MaxSim block")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--gpu", default="6")
    ap.add_argument("--out", default=None)
    ap.add_argument("--submit", action="store_true",
                    help="official test submission: reuse the (alpha,beta) and channel "
                         "weights fitted on a validation split (--hparams) instead of "
                         "re-fitting on the test split, then rank and write the official "
                         "submission file(s). Needs --hparams and --sub-dir.")
    ap.add_argument("--hparams", default=None,
                    help="json written by a previous validation run (--out); supplies "
                         "full_ab / full_all.w / pool_choice")
    ap.add_argument("--sub-dir", default=None, help="output dir for the submission files")
    ap.add_argument("--sub-name", default="relic", help="submission filename prefix")
    ap.add_argument("--fixed-hparams", default=None,
                    help="cross-dataset frozen protocol: read (alpha,beta), channel "
                         "weights w, pooling and CSLS (k,lambda) from a finished run on "
                         "another split and apply them verbatim, skipping every grid "
                         "search, so the number reported here involves no tuning on this "
                         "dataset.")

    # Every other switch in the original research code belonged to an experiment
    # that was falsified and is no longer exposed on the CLI.  The values below
    # are the original defaults, so the shipped behaviour is unchanged.
    ap.set_defaults(
        select="all", pca=0, refpatch=False, ref_k=0, ref_pool=None,
        r3_resid_lam=0.5, r3_sigma=3.0, r3_slots="concat", r3_r_mode="argmax",
        r3_sbs=128, r3_diag=False, r3_role=False,
        r2x=False, r2x_mode="min", dump=None, neg_av=False,
        old_filter="off", drop=None, add=None,
        r5_mode="hub", r5_ms_k=50, r5_xc=0.0, r3_orth=False,
        r5_role=False, r5_role_k=12,
        r5dual=False, r5dual_base="all", r5dual_w_ref=0.05, r5dual_w_top1=0.01,
        r5dual_gate=0.5, r5dual_gate_mode="medrank", r5dual_nz_thr=0.02,
        r5dual_min_n=0, r5dual_k1=20, r5dual_k2=10,
        r5kr=0, r5kr_k2=0, r5kr_anchor="ref", r5role_roles="O,Om,A,ctx",
        r5role_stat="mean", r5kr_alpha=0.5, r5kr_mix=0.0, r5kr_w=None,
        dump_probes=None,
    )
    a = ap.parse_args()
    if a.cqcc:
        # Frozen module-2 (CQCC) unified config.  Set only the fields the user left at
        # their default, so an explicit flag still wins.  One rho drives K on every
        # dataset -> the method is identical across datasets (no per-dataset switch).
        _cqcc = {"r3_mode": "cosdisp_xc", "r5dual": True, "r5dual_base": "all",
                 "r5dual_gate_mode": "nz", "r5dual_gate": 0.5, "r5dual_nz_thr": 0.0,
                 "r5dual_min_n": 0, "r5dual_w_ref": 0.0, "r5dual_w_top1": 0.0,
                 "r5dual_k1": 20, "r5dual_k2": 10, "r5_ms": 0.120,
                 "r5_ms_rho": 0.018, "r5_xc": 0.0}
        _def = {"r3_mode": "off", "r5dual": False, "r5dual_base": "all",
                "r5dual_gate_mode": "medrank", "r5dual_gate": 0.5, "r5dual_nz_thr": 0.02,
                "r5dual_min_n": 0, "r5dual_w_ref": 0.05, "r5dual_w_top1": 0.01,
                "r5dual_k1": 20, "r5dual_k2": 10, "r5_ms": 0.0,
                "r5_ms_rho": 0.0, "r5_xc": 0.0}
        for _k, _v in _cqcc.items():
            if getattr(a, _k) == _def[_k]:
                setattr(a, _k, _v)
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

    import faiss
    import torch

    name = a.dataset
    desc_file, AB0 = DATASETS[name]
    desc_path = a.desc or (ART_DIR / "desc" / desc_file)
    views_path = a.views or (ART_DIR / "views" / ("views_qwen3vl8b_%s.json" % name))
    roles_path = a.roles or (ART_DIR / "roles" / ("roles_qwen3vl8b_%s.json" % name))
    _clip_spec = CLIP_SPECS[a.clip]
    patch_path = Path(a.patch) if a.patch else (
        PATCH_DIR / ("%s_patch%s.npz" % (name, _clip_spec["patch_suffix"])))
    man_path = Path(a.manifest) if a.manifest else (MAN_DIR / ("%s.json" % name))
    idx_dir = Path(a.index_dir) if a.index_dir else (IDX_DIR / name / _clip_spec["idx_tag"])

    man = json.load(open(man_path))
    qs = man["queries"]
    corpus = [str(x) for x in man["corpus_ids"]]
    _, identifiers, _ = load_faiss_index(idx_dir)
    assert [str(x) for x in identifiers] == corpus, "index order != manifest order"
    fi = faiss.read_index(str(idx_dir / "index.faiss"))
    C = norm_rows(fi.reconstruct_n(0, fi.ntotal))
    N = len(corpus)
    id2idx = {c: i for i, c in enumerate(corpus)}

    views = json.load(open(views_path))
    desc_map = json.load(open(desc_path))
    roles = json.load(open(roles_path))

    def fld(i, k):
        return " ".join(str(views.get(str(i), {}).get(k, "") or "").split())

    def rle(i):
        return roles.get(str(i), {}) or {}

    if a.submit:
        # The official submission must contain an entry for *every* test query, so
        # the view-availability filter is not applied here: a query with missing
        # views degrades to the single-vector base path instead of being dropped.
        idx = list(range(len(qs)))
    else:
        idx = [i for i in range(len(qs)) if str(i) in views
               and all(str(views[str(i)].get(k, "")).strip() for k in ("desc", "detail", "rewrite"))]
    n = len(idx)
    is_circo = name.startswith("circo")
    is_cirr = name.startswith("cirr")
    mask_ref = is_circo or is_cirr
    print("dataset=%s corpus=%d queries=%d used=%d" % (name, N, len(qs), n), flush=True)

    def texts_role(i):
        ops = rle(i).get("ops", []) or []
        new = " ".join(str(o.get("new", "") or "") for o in ops if o.get("new"))
        # Phase B3 audit: ``new`` is grounded in the modification caption 85-95%
        # of the time, but ``old`` is traceable to the reference description only
        # ~43% of the time and leaks into the caption 26-36% of the time.  A
        # negative-evidence channel must not be fed wrong negatives, so the
        # grounding test is applied to ``old`` only.
        if a.old_filter != "off":
            refdesc = fld(i, "desc") + " " + fld(i, "detail")
            cap = str(qs[i].get("caption", ""))
            keep = []
            for o in ops:
                od = str(o.get("old", "") or "")
                if not od:
                    continue
                if _ov(od, refdesc) < 1:
                    continue
                if a.old_filter == "clean" and _ov(od, cap) > 0:
                    continue
                keep.append(od)
            old = " ".join(keep)
        else:
            old = " ".join(str(o.get("old", "") or "") for o in ops if o.get("old"))
        pv = rle(i).get("preserve")
        pv = pv if isinstance(pv, str) else " ".join(pv or [])
        return new, old, pv, str(rle(i).get("scene", "") or "")

    TEXTS = {"d0": [], "dv": [], "de": [], "rw": [], "av": [], "rel": [],
             "tg": [], "O": [], "Om": [], "A": [], "ctx": []}
    for i in idx:
        new, old, pv, sc = texts_role(i)
        TEXTS["d0"].append(desc_map.get(str(i)) or fld(i, "desc"))
        TEXTS["dv"].append(fld(i, "desc"))
        TEXTS["de"].append(fld(i, "detail"))
        TEXTS["rw"].append(fld(i, "rewrite"))
        TEXTS["av"].append(fld(i, "avoid"))
        TEXTS["rel"].append(qs[i]["caption"])
        TEXTS["tg"].append(str(rle(i).get("target", "") or ""))
        TEXTS["O"].append(new)
        TEXTS["Om"].append(old)
        TEXTS["A"].append(pv)
        TEXTS["ctx"].append(sc)
    TEXTS = {k: [" ".join(str(t).split()) or "none" for t in v] for k, v in TEXTS.items()}
    coverage = {k: int(sum(1 for t in v if t != "none")) for k, v in TEXTS.items()}
    print("coverage %s" % coverage, flush=True)

    ROLES = [k for k in ("O", "Om", "A", "ctx") if coverage[k] > 0]
    # ``rw``/``de`` are the text views of the single-vector line: keeping them as
    # *channels* lets RELIC subsume M1 instead of competing with it.
    VSETS = ["rw", "de", "O", "Om", "A", "ctx"]
    # ``avoid`` states what the target must NOT contain and has ~98% coverage,
    # far above ``A``/``ctx`` (~55%); it is a natural second negative-evidence
    # channel next to ``Om``, so it may only enter with a non-positive weight.
    NGR = dict(NGRID)
    if a.neg_av:
        VSETS = VSETS + ["av"]
        NGR["av"] = (0.0, -0.02, -0.05, -0.1, -0.2, -0.4)
    # Phase C4 role-necessity ablation: removing a channel from VSETS/ROLES makes
    # the very same RELIC_all row read "RELIC minus that role".
    if a.drop:
        for c in a.drop.split(","):
            c = c.strip()
            if c in VSETS:
                VSETS.remove(c)
            if c in ROLES:
                ROLES.remove(c)
    # Phase C4 control: channels are query-side only, so an extra one is free.
    # ``--add tg`` feeds the whole R1 target description as a *single* channel --
    # the "one good text, no role split" baseline the role decomposition must beat.
    if a.add:
        for c in a.add.split(","):
            c = c.strip()
            if c in TEXTS and c not in VSETS:
                VSETS.append(c)
    ALLSETS = ["d0", "dv", "de", "rw", "av"] + ROLES

    # ---------------------------------------------------------- query encoding --
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        _clip_spec["model"], pretrained=str(_clip_spec["weights"]),
        device="cuda", force_quick_gelu=False)
    model.eval().requires_grad_(False)
    tok = open_clip.get_tokenizer(_clip_spec["model"])
    import torch.nn.functional as F

    def embed(v):
        out = []
        with torch.no_grad():
            for s in range(0, len(v), 256):
                t = tok(v[s:s + 256], context_length=77).cuda()
                out.append(F.normalize(model.encode_text(t).float(), dim=-1).cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32)

    def tokens(v):
        outs, masks = [], []
        with torch.no_grad():
            for s in range(0, len(v), 256):
                t = tok(v[s:s + 256], context_length=77).cuda()
                x = model.token_embedding(t) + model.positional_embedding
                x = model.ln_final(model.transformer(x, attn_mask=model.attn_mask)).float()
                x = x @ model.text_projection
                eot = t.argmax(dim=-1)
                m = torch.zeros_like(t, dtype=torch.bool)
                for r in range(t.shape[0]):
                    m[r, 1:eot[r]] = True
                outs.append(x.cpu().numpy().astype(np.float32))
                masks.append(m.cpu().numpy())
        return np.concatenate(outs, 0), np.concatenate(masks, 0)

    t0 = time.time()
    T = {k: embed(TEXTS[k]) for k in ("d0", "dv", "de", "rw", "av", "rel")}
    # R3's role views are also needed as *dense* vectors, not only as token sets:
    # the merged R5 builds one k-reciprocal probe per role, and a probe has to be a
    # single vector.  Kept in a separate dict so nothing that iterates over T (the
    # single-view table, the channel weights) changes shape.
    TR = {k: embed(TEXTS[k]) for k in ROLES}
    if a.dump_probes:
        # The gallery-side k-reciprocal encoding is the same for every probe, so it
        # is built once offline (protocol/r5_graph_cache.py) and the probe vectors are
        # cached here; all probe variants are then free to evaluate.
        keep = np.array(idx, dtype=np.int64)
        d = {"keep_idx": keep,
             "ref_idx": np.array([id2idx[str(qs[i]["reference"])] for i in idx],
                                 dtype=np.int64)}
        for k in T:
            d["T_" + k] = T[k]
        for k in TR:
            d["TR_" + k] = TR[k]
        # Role-availability mask: a query whose role text is empty is stored as the
        # literal placeholder "none" and embeds to a *shared* constant vector, so its
        # role channel is degenerate and must be maskable offline.  Without this the
        # role-consensus probe would average in ~50% garbage on A/ctx (CIRCO coverage
        # 123/220 and 112/220) and the negative result would be uninterpretable.
        d["role_mask"] = np.array([[TEXTS[r][j] != "none" for r in ROLES]
                                   for j in range(len(idx))], dtype=bool)
        d["role_names"] = np.array(ROLES)
        np.savez_compressed(a.dump_probes, **d)
        print("dumped probes -> %s  (%d queries, %d role views)"
              % (a.dump_probes, len(idx), len(TR)), flush=True)
        return
    TOK, MSK = {}, {}
    for k in VSETS:
        if k in TEXTS:
            TOK[k], MSK[k] = tokens(TEXTS[k])
    timing = {"query_encode_s": time.time() - t0, "n_query_views": len(VSETS)}
    print("query encoding %.1fs" % timing["query_encode_s"], flush=True)

    # -------------------------------------------------- gallery patch streaming --
    z = np.load(patch_path, mmap_mode="r")
    G = z["patch"]                                     # (N, 256, 768) fp16
    P, D = int(G.shape[1]), int(G.shape[2])
    print("patch cache %s  %s" % (patch_path, (G.shape,)), flush=True)

    if a.refpatch and name.startswith(("cirr", "circo")):
        # Reference-image patch channel: the reference is inside the gallery, so its
        # 256 patch tokens come from the same cache at zero extra extraction cost.
        # They act as a *visual* "keep" channel, the patch-level twin of the global
        # f_ref blend -- but unlike f_ref it is not diluted by the CLS pooling, so
        # it can reward a candidate that keeps the same object/attribute as the ref.
        _rid = np.array([id2idx[str(qs[i]["reference"])] for i in idx])
        rt = np.ascontiguousarray(G[_rid]).astype(np.float32)     # (n,256,D)
        rt = F.normalize(torch.from_numpy(rt).cuda(), dim=-1)
        if a.ref_k and 0 < a.ref_k < rt.shape[1]:
            nr = rt.norm(dim=-1)
            keep = nr.topk(a.ref_k, dim=1).indices
            rmask = torch.zeros_like(rt, dtype=torch.bool)
            rmask.scatter_(1, keep.unsqueeze(-1).expand(-1, -1, rt.shape[-1]), True)
            rt = rt.masked_select(rmask).view(rt.shape[0], a.ref_k, rt.shape[2])
        RTOK = rt.float().cpu().numpy()
        del rt
        torch.cuda.empty_cache()
        TOK["R"] = RTOK
        MSK["R"] = np.ones((len(idx), RTOK.shape[1]), dtype=bool)
        VSETS.append("R")
        print("  [refpatch] R channel: %s (ref_k=%d)" % (RTOK.shape, a.ref_k), flush=True)

    # token selection indices (deterministic, computed once)
    K = min(a.tokens, P)
    rnd_idx = None
    if a.select == "random":
        rng = np.random.default_rng(1234)
        rnd_idx = np.sort(rng.choice(P, size=K, replace=False)).astype(np.int64)

    pmean, pcomp = None, None
    if a.pca and a.pca < D:
        print("fitting PCA(%d) on %d gallery patch tokens" % (a.pca, K * min(N, 4000)), flush=True)
        samp = []
        for s in range(0, N, max(1, N // 4000)):
            blk = torch.from_numpy(np.ascontiguousarray(G[s:s + 1])).cuda().float()
            samp.append(blk.reshape(-1, D))
        X = torch.cat(samp, 0)
        Xn = torch.nn.functional.normalize(X, dim=-1)
        pmean = Xn.mean(0)
        Xc = Xn - pmean
        # top-`pca` right singular vectors of the (huge) token matrix
        _, _, Vh = torch.pca_lowrank(Xc, q=a.pca, center=False)
        pcomp = Vh[:, :a.pca].contiguous()          # (D, pca)
        del X, Xn, Xc
        torch.cuda.empty_cache()
        print("  PCA done, kept %s" % (tuple(pcomp.shape),), flush=True)

    def gal_chunk(g0, g1):
        """Load a gallery slice as an fp16 tensor in the shared CLIP space."""
        blk = torch.from_numpy(np.ascontiguousarray(G[g0:g1])).cuda().float()  # (b,P,D)
        blk = torch.nn.functional.normalize(blk, dim=-1)
        ii = None
        if a.select == "last":
            ii = torch.arange(P - K, P, device=blk.device)[None, :].expand(blk.shape[0], -1)
        elif a.select == "random":
            ii = torch.from_numpy(rnd_idx).cuda()[None, :].expand(blk.shape[0], -1)
        elif a.select == "norm":
            ii = blk.norm(dim=-1).topk(K, dim=1).indices
        if ii is not None:
            blk = torch.gather(blk, 1, ii.unsqueeze(-1).expand(-1, -1, D))
        if pcomp is not None:
            blk = (blk - pmean) @ pcomp
            blk = torch.nn.functional.normalize(blk, dim=-1)
        return blk.half()

    def pool_from_best(best, mk, pool, frac, L):
        if pool == "mean":
            nv = mk.sum(1).float()
            sm = best.masked_fill(~mk.unsqueeze(-1), 0.0).sum(1)
            return (sm / nv[:, None]).float().cpu().numpy()
        nv = mk.sum(1).float()
        kk = torch.clamp((nv * frac).ceil().long(), 1, L)
        top = best.topk(int(kk.max().item()), dim=1).values
        sel = (torch.arange(top.shape[1], device=top.device)[None, :, None] < kk[:, None, None])
        sm = (top * sel).sum(1)
        return (sm / kk[:, None].float()).float().cpu().numpy()

    L = 77
    MS = {k: {pn: np.zeros((n, N), dtype=np.float32) for pn in POOLS} for k in VSETS}
    for k in VSETS:
        t = torch.from_numpy(np.ascontiguousarray(TOK[k])).float().cuda()
        if pcomp is not None:
            # query text tokens must live in the same PCA basis as the gallery
            # tokens; gal_chunk normalises *before* projecting, so mirror that.
            t = torch.nn.functional.normalize(t, dim=-1)
            t = (t - pmean) @ pcomp
            t = torch.nn.functional.normalize(t, dim=-1)
        TOK[k] = t.half()
        MSK[k] = torch.from_numpy(MSK[k]).cuda()

    # ------------------------------------------------------ R3 (RLIS) geometry --
    # R3 asks a different question from a MaxSim channel.  A MaxSim channel asks
    # "does the target text match anywhere in g?".  R3 asks "did the *specific
    # change* the caption asks for actually happen, at the place the reference image
    # says it should happen?".  Two ingredients:
    #   * a direction d_k = t_new - t_old per R1 op -- the semantic axis of the edit;
    #   * a location r_k on the reference image -- the reference patch that best
    #     matches the *old* content, i.e. where the edit is supposed to bite.
    # The reference image is then the origin of the measurement, not an item to be
    # penalised: c_k = <p_ref[r_k], d_k> is where the reference sits on that axis, so
    # a candidate only earns transfer credit when it has moved *beyond* the origin,
    # and only near r_k.  Nothing is ever pushed away.
    R3_VALID = None
    r3_state = None
    if a.r3_mode != "off":
        def _prep_text(tn):
            t = torch.from_numpy(np.ascontiguousarray(tn)).float().cuda()
            t = torch.nn.functional.normalize(t, dim=-1)
            if pcomp is not None:
                t = (t - pmean) @ pcomp
                t = torch.nn.functional.normalize(t, dim=-1)
            return t

        def _masked_mean(tok_h, msk_h):
            m = msk_h.unsqueeze(-1).float()
            return torch.nn.functional.normalize(
                (tok_h.float() * m).sum(1) / m.sum(1).clamp(min=1.0), dim=-1)

        if a.r3_slots == "per_op":
            s_old, s_new, s_own = [], [], []
            for j, i in enumerate(idx):
                for o in (rle(i).get("ops", []) or []):
                    od = str(o.get("old", "") or "").strip()
                    nw = str(o.get("new", "") or "").strip()
                    if od and nw:
                        s_old.append(od)
                        s_new.append(nw)
                        s_own.append(j)
            if not s_old:                       # no usable op -> degrade to concat
                s_old, s_new, s_own = TEXTS["Om"], TEXTS["O"], list(range(n))
        else:
            s_old, s_new, s_own = TEXTS["Om"], TEXTS["O"], list(range(n))
        so_t, so_m = tokens(s_old)
        sn_t, sn_m = tokens(s_new)
        with torch.no_grad():
            oldv = _masked_mean(_prep_text(so_t), torch.from_numpy(so_m).cuda())
            newv = _masked_mean(_prep_text(sn_t), torch.from_numpy(sn_m).cuda())
        dslot = (newv - oldv).contiguous()                       # (S, D) directions
        s_own = np.asarray(s_own, np.int64)
        ref_row = np.array([id2idx[str(qs[i]["reference"])] for i in idx])
        S = int(dslot.shape[0])
        dhat = torch.nn.functional.normalize(dslot, dim=-1)      # (S, D) unit axes
        # the per-slot average is linear, so a query's channel collapses to one axis
        s_cnt = np.bincount(s_own, minlength=n).astype(np.float32)
        dbar = np.zeros((n, int(dslot.shape[1])), np.float32)
        np.add.at(dbar, s_own, dhat.cpu().numpy())
        dbar /= np.maximum(s_cnt, 1)[:, None]
        r_pos = np.zeros(S, np.int64)
        r_margin = np.zeros(S, np.float32)
        c_org = np.zeros(S, np.float32)
        R3_NAMES = [m.strip() for m in a.r3_mode.split(",") if m.strip()]
        # one channel per variant, each fitted with its own weight: R3, R3b, R3c, ...
        R3_TAGS = ["R3"] + ["R3" + chr(ord("a") + i) for i in range(1, len(R3_NAMES))]
        # the backward-evidence operand of --r3_mode bidir: <C_g, t_old(q)>.  Must be
        # encoded here, while the text encoder is still alive (it is freed right after
        # the MaxSim pass).
        old_mat = None
        if "bidir" in R3_NAMES:
            to_, mo_ = tokens(TEXTS["Om"])
            with torch.no_grad():
                old_mat = _masked_mean(_prep_text(to_),
                                       torch.from_numpy(mo_).cuda()).float().cpu().numpy()
        r3_state = {"dslot": dslot, "dhat": dhat, "dbar": dbar, "own": s_own, "S": S,
                    "cnt": s_cnt, "ref_row": ref_row, "r_pos": r_pos,
                    "r_margin": r_margin, "old_mat": old_mat,
                    "c_org": torch.from_numpy(c_org).cuda()}
        if a.r3_mode != "off" and any(m in ("soft", "hard") for m in R3_NAMES):
            # spatial variant (kept for the record: the localisation diagnostic below
            # showed a text-derived edit region is worse than chance, so this path is
            # not the recommended one)
            bq = 96
            for q0 in range(0, n, bq):
                q1 = min(q0 + bq, n)
                sl = np.flatnonzero((s_own >= q0) & (s_own < q1))
                if sl.size == 0:
                    continue
                rp = torch.from_numpy(np.ascontiguousarray(G[ref_row[q0:q1]])).cuda().float()
                rp = torch.nn.functional.normalize(rp, dim=-1)
                if pcomp is not None:
                    rp = (rp - pmean) @ pcomp
                    rp = torch.nn.functional.normalize(rp, dim=-1)
                sl_t = torch.from_numpy(sl).cuda()
                rps = rp[torch.from_numpy(s_own[sl] - q0).cuda()]     # (m, P, D)
                match = torch.einsum("mpd,md->mp", rps, oldv[sl_t])
                loc = match.argmax(1)
                r_pos[sl] = loc.cpu().numpy()
                c_org[sl] = torch.einsum(
                    "md,md->m", rps[torch.arange(len(sl), device=loc.device), loc],
                    dslot[sl_t]).float().cpu().numpy()
                r_margin[sl] = (match.max(1).values - match.mean(1)).cpu().numpy()
                del rp, rps, match
                torch.cuda.empty_cache()
            if a.r3_r_mode == "random":                   # localisation control
                r_pos = np.random.default_rng(2026).integers(0, P, size=S)
            elif a.r3_r_mode == "center":
                r_pos = np.full(S, 8 * 16 + 8, np.int64)
            pg = np.arange(P)
            gr, gc = pg // 16, pg % 16                        # 16x16 patch grid
            if P != 256:
                print("  [r3] WARNING: P=%d != 256, the 16x16 grid is an assumption" % P,
                      flush=True)
            d2 = (gr[None, :] - gr[r_pos][:, None]) ** 2 + (gc[None, :] - gc[r_pos][:, None]) ** 2
            _sig = a.r3_sigma if a.r3_sigma > 0 else 1e6
            rand_pos = np.random.default_rng(7).integers(0, P, size=S)
            d2r = (gr[None, :] - gr[rand_pos][:, None]) ** 2 + (gc[None, :] - gc[rand_pos][:, None]) ** 2
            r3_state.update({
                "kern": torch.from_numpy(
                    np.exp(-d2 / (2.0 * _sig ** 2)).astype(np.float32)).cuda(),
                "kern_unif": torch.ones((S, P), dtype=torch.float32, device="cuda"),
                "kern_rand": torch.from_numpy(
                    np.exp(-d2r / (2.0 * _sig ** 2)).astype(np.float32)).cuda(),
                "r_pos": r_pos, "r_margin": r_margin})
            print("R3 geom: slots=%d (%s) r_mode=%s sigma=%.2f r-margin(mean)=%.3f"
                  % (S, a.r3_slots, a.r3_r_mode, a.r3_sigma, float(r_margin.mean())),
                  flush=True)
        else:
            print("R3 geom: slots=%d (%s) modes=%s  ||d||=%.3f"
                  % (S, a.r3_slots, R3_NAMES, float(dslot.norm(dim=-1).mean())),
                  flush=True)

    t0 = time.time()
    for g0 in range(0, N, a.gbs):
        g1 = min(g0 + a.gbs, N)
        GP = gal_chunk(g0, g1)                            # (b,K',d)
        for sb in range(0, g1 - g0, a.sub):
            sb1 = min(sb + a.sub, g1 - g0)
            pb = GP[sb:sb1].reshape(-1, GP.shape[-1])     # (b*K', d)
            for k in VSETS:
                for s in range(0, n, a.qbs):
                    tk = TOK[k][s:s + a.qbs]
                    mk = MSK[k][s:s + a.qbs]
                    lq = tk.shape[1]           # 77 text tokens, 256 for the R channel
                    sc = tk @ pb.T
                    sc = sc.view(tk.shape[0], lq, -1, GP.shape[1]).max(-1).values
                    for pn, (pl, fr) in POOLS.items():
                        MS[k][pn][s:s + tk.shape[0], g0 + sb:g0 + sb1] = pool_from_best(
                            sc, mk, pl, fr, lq)
        if (g0 // a.gbs) % 5 == 0:
            frac = (g1) / float(N)
            eta = (time.time() - t0) / max(frac, 1e-6) * (1 - frac)
            print("  gallery %6d/%d (%.0f%%, eta %.0fs)" % (g1, N, 100 * frac, eta), flush=True)
    timing["maxsim_s"] = time.time() - t0
    timing["maxsim_views"] = len(VSETS)
    print("MaxSim done in %.1fs" % timing["maxsim_s"], flush=True)

    # ---- R2 patch-level hard exclusion (opt-in, `--r2x`) -------------------------
    # O ("must appear") and Om ("must vanish") are mutually exclusive at the *patch*
    # level: one patch cannot at the same time be the evidence that the new content is
    # present and the evidence that the old content is still there.  Any fixed weighted
    # sum scores the two roles independently, so it cannot express this constraint.  We
    # take, per patch p,
    #     x_p = min( max_{t in O} <t,p> , 1 - max_{t in Om} <t,p> )
    # and keep the best patch.  Both terms live on the same cosine scale, so the
    # conjunction reads "new present AND old absent".  Queries missing either role are
    # switched off through VALID below.
    if a.r2x and "O" in TOK and "Om" in TOK:
        X = np.zeros((n, N), np.float32)
        for g0 in range(0, N, a.gbs):
            g1 = min(g0 + a.gbs, N)
            GP = gal_chunk(g0, g1)                          # (b,K',d)
            for sb in range(0, g1 - g0, a.sub):
                sb1 = min(sb + a.sub, g1 - g0)
                pb = GP[sb:sb1].reshape(-1, GP.shape[-1])
                for s in range(0, n, a.qbs):
                    s1 = min(s + a.qbs, n)
                    to, tm = TOK["O"][s:s1], TOK["Om"][s:s1]
                    mo = (~MSK["O"][s:s1].to(torch.bool))[:, :, None, None]
                    mm = (~MSK["Om"][s:s1].to(torch.bool))[:, :, None, None]
                    so = (to @ pb.T).view(s1 - s, to.shape[1], -1, GP.shape[1])
                    sm = (tm @ pb.T).view(s1 - s, tm.shape[1], -1, GP.shape[1])
                    so = so.masked_fill(mo, -1e4).max(1).values     # (q,b,K')
                    sm = sm.masked_fill(mm, -1e4).max(1).values
                    if a.r2x_mode == "margin":
                        x = torch.clamp(so - sm, min=0.0)
                    elif a.r2x_mode == "zmin":
                        soz = (so - so.mean(-1, keepdim=True)) / (
                            so.std(-1, keepdim=True) + 1e-6)
                        smz = (sm - sm.mean(-1, keepdim=True)) / (
                            sm.std(-1, keepdim=True) + 1e-6)
                        x = torch.minimum(soz, -smz)
                    else:
                        x = torch.minimum(so, 1.0 - sm)
                    X[s:s1, g0 + sb:g0 + sb1] = x.max(-1).values.cpu().numpy()
        MS["R2x"] = {"soft": X}
        VSETS.append("R2x")
        print("  R2x[hard exclusion, mode=%s] registered (single channel, pooled as "
              "'soft')" % a.r2x_mode, flush=True)

    for k in VSETS:
        if k in TOK:                       # R2x/R3 are pooled matrices, not text tokens
            TOK[k] = TOK[k].cpu()
    del model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------- ranking data --
    ref_ids = [qs[i]["reference"] for i in idx]
    if is_circo:
        # the test split ships no ground truth, only the val split does
        target_sets = [[str(t) for t in (qs[i].get("targets") or [])] for i in idx]
        tg = [t[0] if t else None for t in target_sets]
    else:
        # test1 ships no ground-truth targets; the empty list becomes None and the
        # label-dependent helpers below are never reached in --submit mode.
        tg = [str(qs[i]["targets"][0]) if qs[i]["targets"] else None for i in idx]
        target_sets = None
    subsets = [[str(m) for m in (qs[i].get("subset") or [])] for i in idx]
    ref_idx = np.array([id2idx[r] for r in ref_ids])
    f_ref = C[ref_idx]
    s_rel = T["rel"] @ C.T
    s_refimg = f_ref @ C.T
    main_key = "mAP@5" if is_circo else "R@1"

    def ref_kreciprocal(k1, k2=0, mix=0.0, probe=None):
        """R5': reference-anchored k-reciprocal re-ranking (Zhong et al., CVPR'17).

        The probe of query q is its *reference image* I_r, which in composed
        retrieval is itself a gallery item, so the whole computation lives in image
        space.  Two consequences matter for the paper:

        * it is a **set** operation on the reference's mutual-neighbour structure
          (Jaccard over neighbour encodings), not a per-column scale, so it is a
          different mechanism from CSLS / hubness correction;
        * it reads **no query rows and no labels**: the encodings depend only on the
          gallery graph and on each reference image, so the result is a gallery-side
          index that can be built once and reused, and no cross-query statistic ever
          sees the held-out rows.  (CSLS's r_g is exactly such a cross-query
          statistic, which is why --csls has to be refitted per fold.)

        The encoding is stored **sparsely**: only the k1+1 nearest neighbours of each
        gallery item are ever read, so a kNN graph replaces the N x N distance matrix
        and the cost is O(N * k1 * k2) instead of O(N^2).  This is a memory change and
        not an approximation -- the neighbour graph is exact and every distance that
        enters an encoding is recomputed from the CLIP features -- which is what lets
        the module run on the 123k-image CIRCO gallery at all.

        Returns a (n, N) score, higher = better.
        """
        k2 = int(np.around(k1 / 2.0)) if k2 <= 0 else int(k2)
        k1q = int(min(k1, N - 1))
        t_kr = time.time()

        # ---- neighbourhood graph: the k1+1 nearest gallery items of every item -----
        # ``sim_kth[j]`` is the similarity of j's k1-th neighbour: it is what decides
        # whether a *query-side* probe q sits inside kNN(j, k1) as well, which the
        # reference-only path never has to ask (the reference is a gallery row).
        if N <= 20000:
            dd = (1.0 - (C @ C.T)).astype(np.float32)
            np.maximum(dd, 0.0, out=dd)
            ir = np.argpartition(dd, k1, axis=1)[:, :k1 + 1].astype(np.int64)
            sim_kth = (1.0 - np.partition(dd, k1, axis=1)[:, k1]).astype(np.float32)
            del dd
        else:
            fi = faiss.IndexFlatIP(C.shape[1])
            fi.add(C)
            sims, ir = fi.search(C, k1 + 1)
            ir = ir.astype(np.int64)
            sim_kth = sims[:, -1].astype(np.float32)
            print("  R5'[k-reciprocal] kNN graph by exact FAISS search (%d items)" % N,
                  flush=True)

        # ---- per-row reciprocal set R(i,k1) + CVPR'17 expansion R*(i,k1), sparse ----
        sups, vals = [], []
        for i in range(N):
            fwd = ir[i]
            back = ir[fwd]
            kre = fwd[np.where((back == i).any(1))[0]]        # R(i, k1)
            kre_set = set(kre.tolist())
            exp = set(kre_set)
            for cand in kre:
                ck = ir[cand, :k2 + 1]
                ck = ck[np.where((ir[ck, :k2 + 1] == cand).any(1))[0]]   # R(cand, k2)
                if len(set(ck.tolist()) & kre_set) > (2.0 / 3.0) * len(ck):
                    exp.update(ck.tolist())
            e = np.fromiter(sorted(exp), dtype=np.int64)
            w = np.exp(-(1.0 - C[i] @ C[e].T).astype(np.float32))
            sups.append(e)
            vals.append(w / w.sum())
        if k2 != 1:                                    # CVPR'17 query expansion
            ns, nv = [], []
            for i in range(N):
                js = ir[i, :k2]
                e = np.concatenate([sups[j] for j in js])
                w = np.concatenate([vals[j] for j in js]) / float(k2)
                u, inv = np.unique(e, return_inverse=True)
                acc = np.zeros(len(u), dtype=np.float32)
                np.add.at(acc, inv, w)
                ns.append(u)
                nv.append(acc)
            sups, vals = ns, nv

        # ---- inverted index over the sparse encodings ------------------------------
        fj = np.concatenate(sups)
        fv = np.concatenate(vals)
        owner = np.repeat(np.arange(N), [len(s) for s in sups])
        order = np.argsort(fj, kind="stable")
        fj, fv, owner = fj[order], fv[order], owner[order]
        uj, st = np.unique(fj, return_index=True)
        en = np.append(st[1:], len(fj))
        print("  R5'[k-reciprocal] encoding built in %.1fs (%d nnz, %.0f MB)"
              % (time.time() - t_kr, len(fj), len(fj) * 12 / 1e6), flush=True)

        # ---- Jaccard between each probe and every gallery item ---------------------
        # both rows sum to 1, so sum(max) = 2 - sum(min) and the Jaccard distance is
        # 1 - sum(min) / (2 - sum(min)).  ``probe=None`` reproduces the reference-
        # anchored channel exactly (the probe is a gallery row, so its encoding is the
        # one already built above); otherwise the probe is a query-side vector, which
        # has to earn its own reciprocal set against the *gallery* graph.
        if probe is None:
            uniq, back_map = np.unique(ref_idx, return_inverse=True)
            probe_pts = C[uniq]
        else:
            uniq = np.arange(probe.shape[0])
            back_map = uniq
            probe_pts = probe
        jac = np.empty((len(uniq), N), dtype=np.float32)
        for ai, pv in enumerate(probe_pts):
            if probe is None:
                enc = (sups[uniq[ai]], vals[uniq[ai]])
            else:
                sq = (pv @ C.T).astype(np.float32)
                iq = np.argpartition(-sq, k1q)[:k1q + 1]
                kre = iq[sq[iq] >= sim_kth[iq]]           # R(q, k1): both directions
                kre_set = set(kre.tolist())
                exp = set(kre_set)
                for cd in kre:
                    ck = ir[cd, :k2 + 1]
                    ck = ck[np.where((ir[ck, :k2 + 1] == cd).any(1))[0]]
                    if len(set(ck.tolist()) & kre_set) > (2.0 / 3.0) * len(ck):
                        exp.update(ck.tolist())
                e = np.fromiter(sorted(exp), dtype=np.int64)
                w = np.exp(-(1.0 - C[e] @ pv).astype(np.float32))
                w = w / w.sum()
                if k2 != 1:
                    # the probe inherits the encodings of its k2 nearest gallery rows,
                    # exactly as CVPR'17 expands a gallery item's own encoding
                    js = iq[:k2]
                    ee = np.concatenate([sups[j] for j in js])
                    ww = np.concatenate([vals[j] for j in js]) / float(k2)
                    u, inv = np.unique(ee, return_inverse=True)
                    acc = np.zeros(len(u), dtype=np.float32)
                    np.add.at(acc, inv, ww)
                    enc = (u, acc)
                else:
                    enc = (e, w)
            if probe is None:
                # Reference-anchored path: sequential accumulation, bit-identical to
                # the shipped channel (its numbers are already on disk).
                tmp = np.zeros(N, dtype=np.float32)
                for jj, vv in zip(enc[0], enc[1]):
                    k = int(np.searchsorted(uj, jj))
                    if k == len(uj) or uj[k] != jj:
                        continue
                    sl = slice(st[k], en[k])
                    tmp[owner[sl]] += np.minimum(vv, fv[sl])
            else:
                # Query-side probe: the same min-sum over the same inverted index, but
                # gathered in one shot.  The probe loop runs once per *query*, so the
                # per-neighbour Python loop -- negligible for the few thousand
                # reference images -- becomes the runtime bottleneck here.
                cols = np.searchsorted(uj, enc[0])
                keep = (cols < len(uj)) & (uj[np.minimum(cols, len(uj) - 1)] == enc[0])
                cols = cols[keep]
                ln = (en[cols] - st[cols]).astype(np.int64) if len(cols) else np.zeros(0, np.int64)
                tot = int(ln.sum())
                if tot:
                    off = np.repeat(st[cols], ln) + (
                        np.arange(tot) - np.repeat(np.cumsum(ln) - ln, ln))
                    tmp = np.bincount(
                        owner[off],
                        weights=np.minimum(
                            np.repeat(np.asarray(enc[1], np.float32)[keep], ln),
                            fv[off]),
                        minlength=N)[:N].astype(np.float32)
                else:
                    tmp = np.zeros(N, dtype=np.float32)
            jac[ai] = 1.0 - tmp / (2.0 - tmp + 1e-9)
        if mix:
            dv = (1.0 - probe_pts @ C.T).astype(np.float32)
            np.maximum(dv, 0.0, out=dv)
            jac = jac * (1.0 - mix) + dv * mix
        return (1.0 - jac)[back_map]

    def metrics(sim, rows=None):
        if rows is None:
            rows = np.arange(sim.shape[0])
        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        sm = sim[rows]
        if is_circo:
            return circo_metrics(sm, corpus, [ref_ids[j] for j in rows],
                                 [target_sets[j] for j in rows], mask_ref)
        if is_cirr:
            if mask_ref:
                return cirr_metrics(sm, corpus, [ref_ids[j] for j in rows],
                                    [tg[j] for j in rows], [subsets[j] for j in rows])
            return cirr_metrics(sm, corpus, [], [tg[j] for j in rows],
                                [subsets[j] for j in rows])
        return fiq_metrics(sm, corpus, [tg[j] for j in rows])

    def fmt(r):
        keys = [k for k in ("mAP@5", "mAP@10", "mAP@25", "mAP@50", "R@1", "R@5",
                            "R@10", "R@50", "R_subset@1", "R_subset@2", "R_subset@3")
                if k in r]
        return " ".join("%s=%.2f" % (k, r[k]) for k in keys)

    def diag_metrics(sim, rows=None):
        """``metrics`` for pre-submission diagnostics only.

        Test splits (cirr_test1 / circo_test) ship without targets, so every
        diagnostic number is undefined there; in ``--submit`` mode those calls
        must degrade to an empty dict instead of aborting the run.  On the val
        splits this is exactly ``metrics`` bit for bit.
        """
        try:
            return metrics(sim, rows)
        except Exception:
            return {}

    # ------------------------------------------------------------ R3 scoring --
    # M_k(q,g) = max_t ( <p_t, d_k> - c_k ) * exp(-||pos(t)-pos(r_k)||^2 / 2 s^2)
    # The Gaussian keeps the measurement local to the edit region; the subtraction of
    # c_k makes the reference image score exactly 0 there, so a candidate only gains
    # when it really moved past the reference along d_k.  Averaged over the slots of a
    # query.  Cheap: one (b*K, D) @ (D, S) matmul per gallery block, no text tokens.
    R3_DIAG = {}

    def r3_matrix(kern_t):
        S = r3_state["S"]
        dsl = r3_state["dslot"]
        corg = r3_state["c_org"]
        own = r3_state["own"]
        acc = np.zeros((n, N), np.float64)
        t0 = time.time()
        for g0 in range(0, N, a.gbs):
            g1 = min(g0 + a.gbs, N)
            GP = gal_chunk(g0, g1)
            pb = GP.reshape(-1, GP.shape[-1]).float()              # (b*K, D)
            b, Kk = GP.shape[0], GP.shape[1]
            for s0 in range(0, S, a.r3_sbs):
                s1 = min(s0 + a.r3_sbs, S)
                pr = (pb @ dsl[s0:s1].t() - corg[s0:s1][None, :]).view(b, Kk, -1)
                pr = pr * kern_t[s0:s1].t()[None, :, :]
                m = pr.max(1).values                               # (b, m)
                np.add.at(acc, (own[s0:s1][:, None],
                                np.arange(g0, g1)[None, :]), m.t().cpu().numpy())
            del GP, pb
        cnt = np.bincount(own, minlength=n).astype(np.float64)
        cnt[cnt == 0] = 1.0
        print("  R3 matrix %.1fs" % (time.time() - t0), flush=True)
        return (acc / cnt[:, None]).astype(np.float32)

    def r3_paired(refproj_np):
        """mean over slots of  max_t ( <p_g[t],dhat_k> - <p_ref[t],dhat_k> ).

        The reference image is the origin *cell by cell*: the same spatial cell must
        sit further along the edit axis in the candidate than it did in the reference.
        Unlike the pooled variants this is a *conjunction* -- the evidence has to come
        from one cell that both lost the old content and gained the new one -- so it is
        not spanned by the reference-similarity and text-MaxSim channels.
        """
        dhat = r3_state["dhat"]
        S = r3_state["S"]
        own = r3_state["own"]
        cnt = r3_state["cnt"].astype(np.float64)
        cnt[cnt == 0] = 1.0                       # queries with no op stay at 0, not nan
        rp_t = torch.from_numpy(refproj_np).cuda()
        acc = np.zeros((n, N), np.float64)
        t0 = time.time()
        for g0 in range(0, N, a.gbs):
            g1 = min(g0 + a.gbs, N)
            GP = gal_chunk(g0, g1)
            pb = GP.reshape(-1, GP.shape[-1]).float()
            b, Kk = GP.shape[0], GP.shape[1]
            for s0 in range(0, S, a.r3_sbs):
                s1 = min(s0 + a.r3_sbs, S)
                pr = (pb @ dhat[s0:s1].t()).view(b, Kk, -1) - rp_t[s0:s1].t()[None, :, :]
                m = pr.max(1).values
                np.add.at(acc, (own[s0:s1][:, None],
                                np.arange(g0, g1)[None, :]), m.t().cpu().numpy())
            del GP, pb
        print("  R3[paired] matrix %.1fs" % (time.time() - t0), flush=True)
        return (acc / cnt[:, None]).astype(np.float32)

    if r3_state is not None:
        pbar = None
        if any(m in ("disp", "cosdisp_p", "gate", "ratio") for m in R3_NAMES):
            pbar = np.zeros((N, int(r3_state["dslot"].shape[1])), np.float32)
            t0 = time.time()
            for g0 in range(0, N, a.gbs):
                g1 = min(g0 + a.gbs, N)
                GP = gal_chunk(g0, g1)
                pbar[g0:g1] = GP.float().mean(1).cpu().numpy()
                del GP
            print("  R3 patch-mean pass %.1fs" % (time.time() - t0), flush=True)
        refproj = None
        if "paired" in R3_NAMES:
            S = r3_state["S"]
            refproj = np.zeros((S, P), np.float32)
            bq = 64
            for q0 in range(0, n, bq):
                q1 = min(q0 + bq, n)
                sl = np.flatnonzero((r3_state["own"] >= q0) & (r3_state["own"] < q1))
                if sl.size == 0:
                    continue
                rp = torch.from_numpy(np.ascontiguousarray(
                    G[r3_state["ref_row"][q0:q1]])).cuda().float()
                rp = torch.nn.functional.normalize(rp, dim=-1)
                if pcomp is not None:
                    rp = (rp - pmean) @ pcomp
                    rp = torch.nn.functional.normalize(rp, dim=-1)
                sl_t = torch.from_numpy(sl).cuda()
                rps = rp[torch.from_numpy(r3_state["own"][sl] - q0).cuda()]   # (m,P,D)
                refproj[sl] = torch.einsum(
                    "mpd,md->mp", rps, r3_state["dhat"][sl_t]).cpu().numpy()
                del rp, rps
                torch.cuda.empty_cache()

        # --- bidirectional / displacement plumbing (C and f_ref are unit-norm rows,
        # so ||C_g - C_ref||^2 = 2(1 - <C_g, C_ref>) and s_refimg is already in hand) --
        crefr = f_ref                                   # (n, D) reference CLS
        old_mat = r3_state["old_mat"]                   # (n, D) unit old-text vectors

        def _dbar():
            return r3_state["dbar"]

        def _cos_cls(dbc, refc, sref):
            """cos( C_g - refc_q , dbc_q ) for every (q, g), refc/sref already permuted."""
            proj = (dbc @ C.T).astype(np.float32)                    # (n, N) <C_g, d_q>
            porg = np.einsum("nd,nd->n", refc, dbc)                  # (n,)
            den = np.sqrt(np.maximum(2.0 * (1.0 - sref), 1e-6))       # (n, N) ||C_g - ref_q||
            return ((proj - porg[:, None]) / den).astype(np.float32)

        def _cos_patch(dbc, refc, refrow):
            prg = (dbc @ pbar.T).astype(np.float32)                   # (n, N) <pbar_g, d_q>
            porg = np.einsum("nd,nd->n", pbar[refrow], dbc)
            pn = (pbar ** 2).sum(1)[None, :]                          # (1, N)
            prn = (pbar[refrow] ** 2).sum(1)[:, None]                 # (n, 1)
            cross = pbar[refrow] @ pbar.T                             # (n, N)
            den = np.sqrt(np.maximum(pn + prn - 2.0 * cross, 1e-6))
            return ((prg - porg[:, None]) / den).astype(np.float32)

        def _fwd(nm):
            rr = r3_state["ref_row"]
            if nm == "cosdisp_p":
                return _cos_patch(_dbar(), pbar[rr], rr)
            return _cos_cls(_dbar(), crefr, s_refimg)

        var_mats = {}
        for nm in R3_NAMES:
            t0 = time.time()
            if nm == "disp":
                # displacement of the whole image along the edit axis, the reference
                # being the origin.  The origin subtraction is a per-row constant, so
                # after the variance-matched z-score it is a no-op; the channel is the
                # axis position <pbar(g), dbar>.
                M = (r3_state["dbar"] @ pbar.T).astype(np.float32)
            elif nm == "cls":
                # same measurement, but on the gallery's CLS embedding C instead of the
                # pooled patch tokens (C is already in memory, so this is free)
                M = (r3_state["dbar"] @ C.T).astype(np.float32)
            elif nm == "paired":
                M = r3_paired(refproj)
            elif nm in ("cosdisp", "cosdisp_xc", "resid", "bidir", "corr"):
                # the image-space displacement C_g - C_ref, projected on the text axis.
                # The reference is a *subtraction operand* here, so it cannot be written
                # as a per-row constant and the z-score has nothing to cancel.
                Fm = _cos_cls(_dbar(), crefr, s_refimg)
                if nm == "cosdisp":
                    M = Fm
                elif nm == "cosdisp_xc":
                    # cross-query contrast: how much *more* this candidate moved along
                    # THIS query's edit axis than along everybody else's.  The second term
                    # is a per-*candidate* statistic, so it survives the row-wise z-score
                    # and is not spanned by any per-query text channel.
                    M = (Fm - Fm.mean(0, keepdims=True)).astype(np.float32)
                elif nm == "resid":
                    lam = a.r3_resid_lam
                    M = (Fm - lam * np.sqrt(np.maximum(0.0, 1.0 - Fm ** 2))).astype(np.float32)
                elif nm == "corr":
                    # auto-correction gate: the "new attribute is present" text evidence
                    # only earns credit when the image actually moved along the edit axis
                    # -- a product of two *existing* channels, so no additive fit of
                    # linear channels can reproduce it
                    tkey = "tg" if "tg" in MS else ("O" if "O" in MS else "rw")
                    tpool = "top50" if ("tg" in MS or "rw" in MS) else "max"
                    zt = zrow(MS[tkey][tpool if tpool in MS[tkey] else "max"].astype(np.float64))
                    M = (zt * np.maximum(zrow(Fm.astype(np.float64)), 0.0)).astype(np.float32)
                else:                                           # bidir
                    s_old = (C @ old_mat.T).T                   # (n, N) <C_g, t_old(q)>
                    ref_old = np.einsum("nd,nd->n", crefr, old_mat)
                    # >0 : the candidate sits *further from the old attribute* than the
                    # reference does, i.e. the old content really went away
                    M = (np.maximum(Fm, 0.0)
                         * np.maximum(zrow(ref_old[:, None] - s_old), 0.0)).astype(np.float32)
            elif nm == "cosdisp_p":
                M = _cos_patch(_dbar(), pbar[r3_state["ref_row"]], r3_state["ref_row"])
            elif nm in ("gate", "ratio"):
                # origin-anchored *monotone* controls: on purpose they should reproduce
                # the plain axis-position ranking, demonstrating that anchoring the
                # origin without a nonlinearity buys nothing
                prg = (r3_state["dbar"] @ pbar.T).astype(np.float32)
                porg = np.einsum("nd,nd->n", pbar[r3_state["ref_row"]], _dbar())
                if nm == "gate":
                    M = np.maximum(prg - porg[:, None], 0.0).astype(np.float32)
                else:
                    M = (prg / (np.abs(porg)[:, None] + 1e-6)).astype(np.float32)
            elif nm in ("soft", "hard"):
                if nm == "hard":
                    kern_use = (r3_state["kern"] > float(np.exp(-0.5))).float() \
                        if a.r3_sigma > 0 else r3_state["kern"]
                else:
                    kern_use = r3_state["kern"]
                M = r3_matrix(kern_use)
                if a.r3_diag:
                    R3_DIAG["%s_no_spatial" % nm] = diag_metrics(r3_matrix(r3_state["kern_unif"]))
                    R3_DIAG["%s_random_r" % nm] = diag_metrics(r3_matrix(r3_state["kern_rand"]))
            else:
                raise SystemExit("unknown --r3_mode %s" % nm)
            var_mats[nm] = M
            R3_DIAG["%s_alone" % nm] = diag_metrics(M)
            print("  R3[%-6s] %.1fs  alone: %s" % (nm, time.time() - t0,
                                                   fmt(R3_DIAG["%s_alone" % nm])), flush=True)
            if a.r3_diag and nm in ("disp", "cls", "paired", "cosdisp", "cosdisp_p",
                                    "resid", "bidir", "gate", "ratio", "corr", "cosdisp_xc"):
                # calibrated null: keep the identical pipeline, swap in another query's
                # edit axis *and* its reference image, so the control isolates the
                # (direction, origin) pair from every other property of the channel
                if nm == "paired":
                    rr = np.random.default_rng(5).permutation(refproj.shape[0])
                    R3_DIAG["%s_null" % nm] = diag_metrics(r3_paired(refproj[rr]))
                elif nm in ("cosdisp", "cosdisp_xc", "resid", "bidir", "corr"):
                    pm = np.random.default_rng(13).permutation(n)
                    Fn = _cos_cls(_dbar()[pm], crefr[pm], s_refimg[pm])
                    if nm == "cosdisp":
                        Mn = Fn
                    elif nm == "cosdisp_xc":
                        Mn = (Fn - Fn.mean(0, keepdims=True)).astype(np.float32)
                    elif nm == "resid":
                        Mn = (Fn - a.r3_resid_lam
                              * np.sqrt(np.maximum(0.0, 1.0 - Fn ** 2))).astype(np.float32)
                    elif nm == "corr":
                        tkey = "tg" if "tg" in MS else ("O" if "O" in MS else "rw")
                        tpool = "top50" if ("tg" in MS or "rw" in MS) else "max"
                        zt = zrow(MS[tkey][tpool if tpool in MS[tkey] else "max"]
                                  .astype(np.float64))
                        Mn = (zt * np.maximum(zrow(Fn.astype(np.float64)), 0.0)).astype(np.float32)
                    else:
                        s_old = (C @ old_mat[pm].T).T
                        ref_old = np.einsum("nd,nd->n", crefr[pm], old_mat[pm])
                        Mn = (np.maximum(Fn, 0.0)
                              * np.maximum(zrow(ref_old[:, None] - s_old), 0.0)).astype(np.float32)
                    R3_DIAG["%s_null" % nm] = diag_metrics(Mn)
                elif nm == "cosdisp_p":
                    pm = np.random.default_rng(13).permutation(n)
                    rr = r3_state["ref_row"][pm]
                    R3_DIAG["%s_null" % nm] = diag_metrics(
                        _cos_patch(_dbar()[pm], pbar[rr], rr))
                elif nm in ("gate", "ratio"):
                    pm = np.random.default_rng(13).permutation(n)
                    db2 = _dbar()[pm]
                    prg = (db2 @ pbar.T).astype(np.float32)
                    porg = np.einsum("nd,nd->n", pbar[r3_state["ref_row"][pm]], db2)
                    Mn = (np.maximum(prg - porg[:, None], 0.0) if nm == "gate"
                          else prg / (np.abs(porg)[:, None] + 1e-6)).astype(np.float32)
                    R3_DIAG["%s_null" % nm] = diag_metrics(Mn)
                else:
                    db2 = r3_state["dbar"][np.random.default_rng(13).permutation(n)]
                    R3_DIAG["%s_null" % nm] = diag_metrics(
                        (db2 @ (pbar if nm == "disp" else C).T).astype(np.float32))

        for tag, nm in zip(R3_TAGS, R3_NAMES):
            MS[tag] = {"soft": var_mats[nm]}
            VSETS.append(tag)
        # late role-necessity drop: R3 tags only come into existence here, so the
        # early `--drop` block cannot reach them.  Re-apply `--drop` after the R3
        # tags are registered so that `--drop R3` genuinely yields the
        # "base without R3" (pure-R5) ablation row.
        if a.drop:
            for _c in a.drop.split(","):
                _c = _c.strip()
                if _c in VSETS:
                    VSETS.remove(_c)
        R3_VALID = (r3_state["cnt"] > 0).astype(np.float32)
        if a.r3_slots == "concat":
            # a query with no old *or* no new content has no direction to measure
            R3_VALID = R3_VALID * (
                (np.array([t != "none" for t in TEXTS["Om"]], dtype=np.float32))
                * (np.array([t != "none" for t in TEXTS["O"]], dtype=np.float32)))
        for tag in R3_TAGS:
            print("  maxsim_alone_%-4s %s" % (tag, fmt(diag_metrics(MS[tag]["soft"]))), flush=True)
        for tag, mm in R3_DIAG.items():
            print("  r3_diag %-14s %s" % (tag, fmt(mm)), flush=True)

    def valid_vec(k):
        """Per-query availability of a channel (0 where there is nothing to say)."""
        if k in TEXTS:
            return np.array([t != "none" for t in TEXTS[k]], dtype=np.float32)
        if k.startswith("R3") and R3_VALID is not None:
            return R3_VALID
        return np.ones(n, dtype=np.float32)

    def col_scale(sim, sel, k):
        """r_g(g) = mean of the k largest scores in column g, over rows ``sel``.

        Cross-query, label-free, and *not* a row-wise transform: it removes the
        component a gallery item shares with many unrelated queries (hubness),
        which is exactly what a late-interaction fusion injects and what a
        per-query scalar patch (P3) cannot see.
        """
        sub = sim[sel]
        kk = min(k, len(sel) - 1)
        if a.r5_mode == "ref":
            # discriminative variant: penalise the part of a column that the *references*
            # -- which are shared across all queries -- already explain, instead of the
            # raw top-k mass.  Rescaled to the spread of the sim column means so the
            # same lam stays comparable.
            sub_sel = np.asarray(sel)
            c = s_refimg[sub_sel].mean(0, keepdims=True)
            s = sim[sub_sel].mean(0, keepdims=True)
            return (c * (s.std() / (c.std() + 1e-9))).astype(np.float32)
        return np.partition(sub, -kk, axis=0)[-kk:, :].mean(0, keepdims=True)

    def csls_apply(sim, sel, k, lam):
        sel = np.asarray(sel)
        if sel.dtype == bool:
            sel = np.flatnonzero(sel)
        # Plain single-scale CSLS.  The multi-scale term lives in the module-2 block
        # below, so the `*_csls` rows stay the honest "pure R5" ablation to beat.
        out = sim - lam * col_scale(sim, sel, k)
        return out.astype(np.float32)

    def calz(k, sel=None):
        """z-scored channel ``k``, optionally role-wise calibrated (``--r5_role``).

        With the flag off this is exactly ``zrow(MSg[k])``, so every existing number
        is reproduced bit for bit.  With it on, each channel loses its own cross-query
        column mass before it enters the fusion: the hubness correction R5 applies once
        to the fused score is here applied per role, so a channel whose gallery columns
        are shared by many unrelated queries is cleaned before being weighted.
        """
        z = zrow(MSg[k])
        if a.r5_role:
            s = np.arange(n) if sel is None else np.asarray(sel)
            z = (z - a.csls_lam * col_scale(z, s, a.r5_role_k)).astype(np.float64)
        return z

    def perq(sim, rows=None):
        """Per-query value of the headline metric (paired tests need the vector)."""
        if rows is None:
            r = np.arange(sim.shape[0])
        else:
            rows = np.asarray(rows)
            r = np.flatnonzero(rows) if rows.dtype == bool else rows
        s = np.array(sim[r], copy=True)
        if mask_ref:
            s[np.arange(len(r)), [id2idx[ref_ids[j]] for j in r]] = -np.inf
        order = np.argsort(-s, axis=1)
        if is_circo:
            out = np.zeros(len(r), dtype=np.float32)
            for i, j in enumerate(r):
                gi = np.array([id2idx[t] for t in target_sets[j] if t in id2idx],
                              dtype=np.int64)
                lab = np.isin(order[i], gi).astype(np.float32)
                prec = np.cumsum(lab) * lab / np.arange(1, len(lab) + 1)
                out[i] = prec[:5].sum() / min(len(target_sets[j]), 5)
            return out * 100.0
        tgt = np.array([id2idx[tg[j]] for j in r], dtype=np.int64)
        return (order[:, 0] == tgt).astype(np.float32) * 100.0

    def score(q, alpha, beta, gamma=0.0):
        fus = norm_rows((1 - alpha) * q + alpha * f_ref)
        sim = (1 + beta) * (fus @ C.T) - beta * (q @ C.T - s_rel)
        if gamma:
            sim = sim - gamma * s_refimg
        return sim

    res = {"dataset": name, "clip": a.clip,
           "n_queries": len(qs), "n_gallery": N, "used": n,
           "coverage": coverage, "roles": ROLES, "folds": a.folds,
           "tokens": {"K": K, "select": a.select, "pca": a.pca},
           "channels": list(VSETS), "neg_av": bool(a.neg_av),
           "old_filter": a.old_filter, "drop": a.drop, "add": a.add,
           "r3": None if r3_state is None else {
               "mode": a.r3_mode, "sigma": a.r3_sigma, "slots": a.r3_slots,
               "r_mode": a.r3_r_mode, "orth": bool(a.r3_orth),
               "n_slots": int(r3_state["S"]),
               "r_margin_mean": float(r3_state["r_margin"].mean()),
               "c_origin_mean": float(r3_state["c_org"].mean())},
           "r3_diag": {k: v for k, v in R3_DIAG.items()},
           "storage_gb": float(N * K * (a.pca or D) * 2 / 1e9)}

    # ------------------------------------------------- official test submission --
    # The standard ZS-CIR protocol selects hyper-parameters on a validation split
    # and reports on the test split.  In --submit mode we do exactly that: the
    # channel weights and (alpha,beta) are read from a finished validation run
    # instead of being re-fitted here, so the test split never tunes anything.
    if a.submit:
        hp = json.load(open(a.hparams))
        al, be = hp["full_ab"]["alpha"], hp["full_ab"]["beta"]
        w = hp["full_all"]["w"]
        gamma = float(hp["full_all"].get("gamma", 0.0))
        pc = hp["pool_choice"]
        MSg = {k: MS[k][pc[k]] for k in VSETS if k in pc}
        sets = [k for k in VSETS if k in w and k in MSg]
        vl = {k: valid_vec(k) for k in sets}
        base = score(T["d0"], al, be)
        sd = base.std(1, keepdims=True)
        sF = base + sum(w[k] * calz(k) * sd * vl[k][:, None] for k in sets)
        if gamma:
            sF = sF - gamma * s_refimg
        s_raw = sF                       # == full_all, the fusion *before* CSLS
        if a.csls:
            # r_g is a per-gallery-column statistic: it lives in the test gallery's
            # own column geometry and cannot be transferred from the val index, so
            # it is estimated from the test queries themselves (plain CSLS).
            sF = csls_apply(sF, np.arange(n), a.csls, a.csls_lam)
        # ---- module 2 (CQCC): multi-scale column term + gated dual anchors --------
        # Mirrors the main-flow block ("R5 (merged, dual-anchor)") bit for bit: the
        # two hubness scales share one operator x -> x - lam*col_scale(x,k); baseS is
        # the CSLS-corrected fusion and raw is the same matrix before CSLS.  No label
        # and no test-split fit enters here -- every parameter comes from the frozen
        # val hparams + the --cqcc flag, exactly as on the val splits.
        if a.r5dual:
            if not a.csls:
                raise SystemExit("--r5dual in --submit needs --csls")
            if a.r5dual_base != "all":
                raise SystemExit("--submit supports --r5dual_base all only")
            baseS = sF.astype(np.float32)
            raw = s_raw.astype(np.float32)
            # Module 2 is anchor-free: when both anchor weights are zero the gallery-side
            # k-reciprocal graph is not built at all (it is the expensive term).
            use_anch = bool(a.r5dual_w_ref) or bool(a.r5dual_w_top1)
            if use_anch:
                kr_ref, kr_top1, medrank = r5dual_channels(
                    baseS, C, ref_idx, n, N, name, a.r5dual_k1, a.r5dual_k2)
                g, gauge = r5dual_gate_value(kr_ref, medrank, N, a.r5dual_gate_mode,
                                             a.r5dual_gate, a.r5dual_nz_thr,
                                             a.r5dual_min_n)
            else:
                kr_ref = kr_top1 = medrank = None
                g, gauge = 1.0, 0.0
            mod = baseS
            ms_k = a.r5_ms_k
            if a.r5_ms_rho > 0:
                _N = raw.shape[1]
                ms_k = int(min(max(round(a.r5_ms_rho * _N), 20), _N // 50))
            if a.r5_ms and ms_k * 50 <= raw.shape[1]:
                mod = (mod - a.r5_ms * col_scale(raw, np.arange(n), ms_k)
                       ).astype(np.float32)
            sd2 = mod.std(1, keepdims=True)
            if use_anch:
                anch = (a.r5dual_w_ref * sd2 * zrow(kr_ref)
                        + a.r5dual_w_top1 * sd2 * zrow(kr_top1))
            else:
                anch = np.float32(0.0)
            xc = np.float32(0.0)
            if a.r5_xc and "R3" in MSg:
                ch3 = calz("R3") * valid_vec("R3")[:, None]
                if a.r3_orth:
                    basis = [zrow(mod)]
                    for kk in VSETS:
                        if kk != "R3":
                            basis.append(calz(kk) * valid_vec(kk)[:, None])
                    if use_anch:
                        basis += [zrow(kr_ref), zrow(kr_top1)]
                    if a.r5_ms and ms_k * 50 <= raw.shape[1]:
                        basis.append(zrow(col_scale(raw, np.arange(n), ms_k)))
                    ch3 = orth_resid(ch3, basis)
                xc = (a.r5_xc * sd2 * ch3).astype(np.float32)
            sF = (mod + g * anch + xc).astype(np.float32)
            print("submit[CQCC]: gate=%.0f gauge=%.4f ms=%.3f ms_k=%d"
                  % (g, gauge, a.r5_ms, ms_k), flush=True)
            try:                          # val splits only; test GT is withheld
                print("submit[CQCC] metrics: %s" % fmt(metrics(sF)), flush=True)
            except Exception:
                pass
        print("submit: ab=(%.2f,%.2f) gamma=%.2f w=%s csls=%d/%.2f sets=%s"
              % (al, be, gamma, {k: w[k] for k in sets}, a.csls, a.csls_lam, sets),
              flush=True)

        order = np.argsort(-sF, axis=1)
        ranked = np.array(corpus, dtype=object)[order]
        ref_col = np.array([str(r) for r in ref_ids], dtype=object)[:, None]
        ranked = ranked[ranked != ref_col].reshape(n, N - 1)

        out = Path(a.sub_dir)
        out.mkdir(parents=True, exist_ok=True)

        if is_circo:
            # CIRCO's server wants a flat {"<query id>": [gallery ids]} mapping:
            # integer query keys, integer image ids, top-50, no wrapper keys and
            # every one of the 800 test queries present.
            sub = {}
            for r in range(n):
                qid = str(int(qs[idx[r]]["qid"]))
                sub[qid] = [int(x) for x in ranked[r].tolist()[:50]]
            f1 = out / ("%s_circo_test_submission.json" % a.sub_name)
            json.dump(sub, open(f1, "w"), sort_keys=True)
            print("wrote %s (%d keys, %d each)" % (f1, len(sub), len(sub["0"])),
                  flush=True)
            res["submit"] = {"hparams": str(a.hparams), "ab": [al, be], "w": w,
                             "gamma": gamma, "csls": [a.csls, a.csls_lam],
                             "files": [str(f1)]}
        else:
            sub = {"version": "rc2", "metric": "recall"}
            gsub = {"version": "rc2", "metric": "recall_subset"}
            for r in range(n):
                qid = str(qs[idx[r]]["qid"])
                row = ranked[r].tolist()
                sub[qid] = row[:50]
                members = {m for m in subsets[r] if m != str(ref_ids[r])}
                gsub[qid] = [x for x in row if x in members][:3]

            f1 = out / ("%s_cirr_ViT-L-14_test_submission.json" % a.sub_name)
            # the leaderboard sample uses the "subset_" prefix *before* the method name
            f2 = out / ("subset_%s_cirr_ViT-L-14_test_submission.json" % a.sub_name)
            json.dump(sub, open(f1, "w"), sort_keys=True)
            json.dump(gsub, open(f2, "w"), sort_keys=True)
            print("wrote %s (%d keys, %d each)"
                  % (f1, len(sub) - 2, len(sub[str(qs[idx[0]]["qid"])])), flush=True)
            print("wrote %s (%d keys)" % (f2, len(gsub) - 2), flush=True)
            res["submit"] = {"hparams": str(a.hparams), "ab": [al, be], "w": w,
                             "gamma": gamma, "csls": [a.csls, a.csls_lam],
                             "files": [str(f1), str(f2)]}
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)
            print("wrote", a.out, flush=True)
        return

    # ------------------------------------------------ frozen hyper-parameters ----
    # Cross-dataset protocol: every hyper-parameter is selected *once* on the CIRR
    # validation split and then applied verbatim here.  Nothing below is searched on
    # this dataset, so the number reported here cannot be inflated by tuning on the
    # very split it is measured on.
    if a.fixed_hparams:
        hp = json.load(open(a.fixed_hparams))
        al, be = hp["full_ab"]["alpha"], hp["full_ab"]["beta"]
        w = hp["full_all"]["w"]
        gamma = float(hp["full_all"].get("gamma", 0.0))
        pc = hp["pool_choice"]
        cs = hp.get("full_all_csls") or hp.get("full_role_csls") or {}
        k = a.csls or int(cs.get("k", 0))
        lam = float(cs.get("lam", a.csls_lam))
        MSg = {kk: MS[kk][pc[kk]] for kk in VSETS if kk in pc}
        sets = [kk for kk in VSETS if kk in w and kk in MSg]
        vl = {kk: valid_vec(kk) for kk in sets}
        base = score(T["d0"], al, be)
        sd = base.std(1, keepdims=True)
        sF = base + sum(w[kk] * calz(kk) * sd * vl[kk][:, None] for kk in sets)
        if gamma:
            sF = sF - gamma * s_refimg
        out = {"dataset": name, "frozen_from": str(a.fixed_hparams),
               "ab": [al, be], "w": {kk: w[kk] for kk in sets}, "gamma": gamma,
               "pool": {kk: pc[kk] for kk in sets}}
        m = metrics(sF)
        out["metrics_no_r5"] = m
        print("frozen ab=(%.2f,%.2f) w=%s pool=%s -> %s"
              % (al, be, out["w"], out["pool"], fmt(m)), flush=True)
        if k:
            m = metrics(csls_apply(sF, np.arange(n), k, lam))
            out["csls"] = [k, lam]
            out["metrics_r5_csls"] = m
            print("  +R5(csls k=%d lam=%.2f) -> %s" % (k, lam, fmt(m)), flush=True)
        if a.r5dual:
            # The gallery-side graph lives in the *target* gallery (like r_g), so
            # it is rebuilt here; only the two weights and the gate threshold are
            # transferred from the CIRR val run.
            baseS = csls_apply(sF, np.arange(n), k, lam) if k else sF
            t0 = time.time()
            # Anchor-free module 2: skip the (expensive) k-reciprocal graph when both
            # anchor weights are zero, keeping only the coarse-scale calibration.
            use_anch = bool(a.r5dual_w_ref) or bool(a.r5dual_w_top1)
            if use_anch:
                kr_ref, kr_top1, medrank = r5dual_channels(
                    baseS, C, ref_idx, n, N, name, a.r5dual_k1, a.r5dual_k2)
                g, gauge = r5dual_gate_value(kr_ref, medrank, N, a.r5dual_gate_mode,
                                             a.r5dual_gate, a.r5dual_nz_thr,
                                             a.r5dual_min_n)
                sd = baseS.std(1, keepdims=True)
                sD = (baseS + g * (a.r5dual_w_ref * sd * zrow(kr_ref)
                                   + a.r5dual_w_top1 * sd * zrow(kr_top1))).astype(np.float32)
            else:
                kr_ref = kr_top1 = medrank = None
                g, gauge = 1.0, 0.0
                sD = baseS.astype(np.float32)
            out["r5dual"] = {"w_ref": a.r5dual_w_ref, "w_top1": a.r5dual_w_top1,
                             "gate_thr": a.r5dual_gate,
                             "gate_mode": a.r5dual_gate_mode, "gauge": gauge,
                             "medrank": medrank,
                             "gate": g, "k1": a.r5dual_k1, "k2": a.r5dual_k2}
            m = metrics(sD)
            out["metrics_r5dual"] = m
            print("  R5[dual] %s=%.4f -> gate=%.0f  (%.0fs) | -> %s"
                  % (a.r5dual_gate_mode, gauge, g, time.time() - t0, fmt(m)),
                  flush=True)
        out["metrics"] = m
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(a.out, "w"), indent=1)
            print("wrote", a.out, flush=True)
        return

    S = {k: score(T[k], AB0[0], AB0[1]) for k in ("d0", "dv", "de", "rw", "av")}
    res["single"] = {"base_ab0_d0": metrics(S["d0"])}
    print("\n=== single views (a=%.2f b=%.2f) ===" % AB0, flush=True)
    for k in ("d0", "dv", "de", "rw", "av"):
        print("  single_%-3s %s" % (k, fmt(metrics(S[k]))), flush=True)
    res["single"]["base_ab0_d0"] = metrics(S["d0"])

    # pooling pick: MaxSim-alone on the full split, per role
    print("\n=== pooling pick (MaxSim-alone %s) ===" % main_key, flush=True)
    res["pooling"] = {}
    pool_choice = {}
    for k in VSETS:
        if len(MS[k]) < len(POOLS):        # e.g. R3: a single, already-pooled channel
            r = {p: metrics(MS[k][p])[main_key] for p in MS[k]}
            res["pooling"][k] = r
            pool_choice[k] = max(r, key=r.get)
            print("  %-3s %s -> %s" % (k, " ".join("%s:%.2f" % (p, v) for p, v in r.items()),
                                       pool_choice[k]), flush=True)
            continue
        r = {pn: metrics(MS[k][pn])[main_key] for pn in POOLS}
        res["pooling"][k] = r
        pool_choice[k] = max(r, key=r.get)
        if k == "R" and a.ref_pool:
            pool_choice[k] = a.ref_pool
        print("  %-3s %s -> %s" % (k, " ".join("%s:%.2f" % (p, v) for p, v in r.items()),
                                   pool_choice[k]), flush=True)
    # per-role pooling is free (a query-side reduction), so keep the best one
    global_pool = max(POOLS, key=lambda pn: np.mean(
        [res["pooling"][k][pn] for k in VSETS if pn in res["pooling"][k]]))
    res["pool_choice"] = pool_choice
    res["global_pool"] = global_pool
    print("  global pooling = %s ; per-role = %s" % (global_pool, pool_choice), flush=True)
    MSg = {k: MS[k][pool_choice[k]] for k in VSETS}
    # ---- R3 role-axis channel (opt-in, `--r3_role`) ------------------------------
    # Unlike `--r3_mode` (a CLS-level cosine against a *text*-space axis), this measures
    # the displacement on the R2 role channels themselves: how far the candidate moved,
    # relative to its own reference image, toward "must-add" (O) and away from
    # "must-remove" (Om).  The component of that displacement explained by *preserve*
    # drift (A) is projected out -- per query, across candidates -- so a candidate that
    # merely changed a lot everywhere earns nothing.  The cross-query column contrast is
    # kept: it is the only part no per-query text channel can span.
    if a.r3_role:
        if all(c in MSg for c in ("O", "Om", "A")):
            rr = np.array([id2idx[str(qs[i]["reference"])] for i in idx])
            ar = np.arange(n)
            dO = (MSg["O"] - MSg["O"][ar, rr][:, None]).astype(np.float64)
            dM = (MSg["Om"] - MSg["Om"][ar, rr][:, None]).astype(np.float64)
            dA = (MSg["A"] - MSg["A"][ar, rr][:, None]).astype(np.float64)
            edit = dO - dM                       # toward new AND away from old
            beta = ((edit * dA).sum(1, keepdims=True)
                    / ((dA * dA).sum(1, keepdims=True) + 1e-9))    # proj_A(edit)
            edit_p = edit - beta * dA            # orthogonal to the preserve drift
            MSg["R3r"] = (edit_p - edit_p.mean(0, keepdims=True)).astype(np.float32)
            VSETS.append("R3r")
            print("  R3[role-axis] R3r: mean|beta|=%.3f  alone: %s"
                  % (float(np.abs(beta).mean()), fmt(metrics(MSg["R3r"]))), flush=True)
        else:
            print("  R3[role-axis] skipped: needs O/Om/A channels (have %s)"
                  % sorted(MSg), flush=True)
    res["maxsim_alone"] = {k: metrics(MSg[k]) for k in VSETS}
    for k in VSETS:
        print("  maxsim_alone_%-3s %s" % (k, fmt(res["maxsim_alone"][k])), flush=True)

    VALID = {k: valid_vec(k) for k in VSETS}
    if "R3r" in VALID:
        # nothing to measure when the query has no old / new / preserve content
        VALID["R3r"] = (np.array([t != "none" for t in TEXTS["O"]], np.float32)
                        * np.array([t != "none" for t in TEXTS["Om"]], np.float32)
                        * np.array([t != "none" for t in TEXTS["A"]], np.float32))
    if "R2x" in VALID:
        # the patch-level exclusion needs both the "must appear" and the "must vanish"
        # role to have content -- otherwise one side is all-masked and the min() is void
        VALID["R2x"] = (np.array([t != "none" for t in TEXTS["O"]], np.float32)
                        * np.array([t != "none" for t in TEXTS["Om"]], np.float32))

    # ------------------------------------------------------- full-split fitting --
    # Headline numbers: everything fitted on the whole split (same protocol as
    # tune_lateint_cirr.py), so they are comparable to the reported 35.02 -> 37.13.
    print("\n=== full-split fit ===", flush=True)

    def ascend(base, sets, valid, grid=WGRID, iters=4, gamma=None):
        sd = base.std(1, keepdims=True)
        CH = {k: calz(k) * sd * valid[k][:, None] for k in sets}
        def fus(w, g=0.0):
            d = sum(w[k] * CH[k] for k in sets)
            return base + d - g * s_refimg if g else base + d
        w = {k: 0.1 for k in sets}
        best = metrics(fus(w))[main_key]
        for _ in range(iters):
            for nm in sets:
                cand = {}
                for v in NGR.get(nm, grid):
                    w2 = dict(w)
                    w2[nm] = v
                    cand[v] = metrics(fus(w2))[main_key]
                w[nm] = max(cand, key=cand.get)
            cur = metrics(fus(w))[main_key]
            if cur <= best + 1e-9:
                best = cur
                break
            best = cur
        g = 0.0
        if gamma is not None:
            cand = {}
            for v in gamma:
                cand[v] = metrics(fus(w, v))[main_key]
            g = max(cand, key=cand.get)
        return w, g, fus(w, g)

    al0, be0 = AB0
    b0 = (-1e9, None, None)
    for al, be in ABGRID:
        r = metrics(score(T["d0"], al, be))[main_key]
        if r > b0[0]:
            b0 = (r, al, be)
    Bal = score(T["d0"], b0[1], b0[2])
    print("  tuned ab = (%.2f, %.2f) -> %s" % (b0[1], b0[2], fmt(metrics(Bal))),
          flush=True)
    res["full_ab"] = {"alpha": b0[1], "beta": b0[2], "metrics": metrics(Bal)}

    fullsim = {}
    for tag, sets in (("role", [k for k in ROLES]), ("all", list(VSETS))):
        if not sets:
            continue
        w, g, sF = ascend(Bal, sets, VALID, iters=5, gamma=G_REF)
        fullsim[tag] = sF
        res["full_" + tag] = {"w": {k: round(v, 3) for k, v in w.items()},
                              "gamma": g, "metrics": metrics(sF)}
        print("  RELIC_%-4s w=%s gamma=%.2f -> %s"
              % (tag, {k: round(v, 2) for k, v in w.items()}, g, fmt(metrics(sF))),
              flush=True)

    # R5: cross-query gallery-side calibration (CSLS-style hubness correction).
    # Applied *after* the weights, because it is a property of the gallery column
    # geometry, not of the per-query combination. Full split -> all rows give r_g.
    if a.csls:
        for tag in list(fullsim):
            sC = csls_apply(fullsim[tag], np.arange(n), a.csls, a.csls_lam)
            fullsim[tag + "_csls"] = sC
            res["full_" + tag + "_csls"] = {"k": a.csls, "lam": a.csls_lam,
                                            "metrics": metrics(sC)}
            print("  +R5(csls k=%d lam=%.2f) on RELIC_%-4s -> %s"
                  % (a.csls, a.csls_lam, tag, fmt(metrics(sC))), flush=True)

    # ---- R5 (merged, dual-anchor): gallery-scale-adaptive neighbourhood ----------
    # The two k-reciprocal probes fail on OPPOSITE sides of the gallery-size axis
    # (measured, protocol/out/relic/r5_gauge_diag.log):
    #   * reference anchor : headAUC 0.564 at N=2.3k (CIRR, +0.67) but 0.556 at
    #     N=123k (CIRCO, +0.00) -- at 123k the reference's mutual-neighbour set is
    #     uninformative, so the channel enters the fusion as noise and a frozen
    #     weight cannot know it (it costs -3.5 mAP@5 at w=0.02);
    #   * retrieval anchor : the probe is the base's OWN top-1, so at large N, where
    #     the base top-1 is often wrong, its neighbourhood is genuinely new
    #     information (+1.45 mAP@5 on CIRCO), while at small N it merely re-encodes
    #     the base and is exactly neutral on R@1.
    # One module uses both anchors.  The reference anchor is scaled by a LABEL-FREE
    # consistency gate read off the target gallery alone:
    #     g = 1[ median_rank( base top-1 inside kr_ref ) / N <= --r5dual_gate ]
    # which is 0.004 (on) on CIRR/FIQ and 0.689 (off) on CIRCO.  No labels and no
    # target-split fitting enter the gate; the weights are frozen on CIRR val.
    r5dual_mats = {}
    if a.r5dual:
        btag = a.r5dual_base
        bkey = btag + "_csls" if a.csls else btag
        if bkey not in fullsim:
            raise SystemExit("--r5dual: %s not built (need --csls for the _csls key)"
                             % bkey)
        baseS = fullsim[bkey]
        t0 = time.time()
        # Module 2 no longer uses the two anchor channels.  Their marginal contribution
        # is small (mean +0.19 over six cells vs +0.22 for the coarse scale) and the
        # reference anchor contradicts the edit-difference story: R1/A penalise
        # candidates that look like the reference, while kr_ref rewards exactly those.
        # When both weights are zero the gallery-side k-reciprocal graph is skipped --
        # it is by far the most expensive term (1064 s on the 123k CIRCO gallery).
        use_anch = bool(a.r5dual_w_ref) or bool(a.r5dual_w_top1)
        if use_anch:
            kr_ref, kr_top1, medrank = r5dual_channels(
                baseS, C, ref_idx, n, N, name, a.r5dual_k1, a.r5dual_k2)
            g, gauge = r5dual_gate_value(kr_ref, medrank, N, a.r5dual_gate_mode,
                                         a.r5dual_gate, a.r5dual_nz_thr,
                                         a.r5dual_min_n)
            r5dual_mats = {"kr_ref": kr_ref.astype(np.float32),
                           "kr_top1": kr_top1.astype(np.float32)}
        else:
            kr_ref = kr_top1 = medrank = None
            g, gauge = 1.0, 0.0
            print("  R5[dual] anchors off (w_ref=w_top1=0): k-reciprocal graph skipped",
                  flush=True)
        # ---- module 2 = R5 (cross-query column calibration) + R3 (direction axis) ----
        # R5 owns the cross-query column operator x - lam*r_g^(k).  Module 2 stacks two
        # further *label-free* column operators on top of it:
        #   (a) a second, coarser hubness scale (multi-scale), and
        #   (b) the R3 editing-direction channel, cross-query column centred, injected
        #       through the same anchor block;
        # both ride the label-free service interval g, so on a gallery whose reference
        # neighbourhood is uninformative (CIRCO, g=0) they switch off and the
        # multi-scale term alone carries the module.
        mod = baseS
        raw = fullsim[btag]      # the fused score *before* the CSLS correction
        # Unified module-2 rule: one rho is valid across every dataset because the
        # coarse scale is derived from the gallery size alone,
        #   k = clip(round(rho*N), 20, N//50),
        # which always satisfies the k*50 <= N validity guard below.
        ms_k = a.r5_ms_k
        if a.r5_ms_rho > 0:
            _N = raw.shape[1]
            ms_k = int(min(max(round(a.r5_ms_rho * _N), 20), _N // 50))
        if a.r5_ms and ms_k * 50 <= raw.shape[1]:
            # the coarse scale is read off the *raw* fused score, so both scales
            # share one operator and differ only in k
            mod = (mod - a.r5_ms * col_scale(raw, np.arange(n), ms_k)
                   ).astype(np.float32)
        sd = mod.std(1, keepdims=True)
        if use_anch:
            anch = (a.r5dual_w_ref * sd * zrow(kr_ref)
                    + a.r5dual_w_top1 * sd * zrow(kr_top1))
        else:
            anch = np.float32(0.0)
        # The *direction axis* (R3) is a query-side, label-free channel: it is NOT
        # reference-conditioned, so it is exempt from the reference-anchor service
        # gate.  Only the kr_ref/kr_top1 anchor block is gated (that is the block
        # whose "reference-consistent => correct" assumption breaks on CIRR).
        xc = np.float32(0.0)
        if a.r5_xc and "R3" in MSg:
            ch3 = calz("R3") * VALID["R3"][:, None]
            if a.r3_orth:
                basis = [zrow(mod)]
                for kk in VSETS:
                    if kk != "R3":
                        basis.append(calz(kk) * VALID[kk][:, None])
                if use_anch:
                    basis += [zrow(kr_ref), zrow(kr_top1)]
                if a.r5_ms and ms_k * 50 <= raw.shape[1]:
                    basis.append(zrow(col_scale(raw, np.arange(n), ms_k)))
                ch3 = orth_resid(ch3, basis)
                print("  R3[orth] %d-channel residual: std=%.3f (raw %.3f) alone: %s"
                      % (len(basis), float(ch3.std()),
                         float((calz("R3") * VALID["R3"][:, None]).std()),
                         fmt(metrics(ch3))), flush=True)
            xc = (a.r5_xc * sd * ch3).astype(np.float32)
        cand = {bkey: baseS}
        cand["dual_nogate"] = (mod + anch + xc).astype(np.float32)
        cand["dual"] = (mod + g * anch + xc).astype(np.float32)
        res["r5dual"] = {"base": bkey, "w_ref": a.r5dual_w_ref,
                         "w_top1": a.r5dual_w_top1, "gate_thr": a.r5dual_gate,
                         "gate_mode": a.r5dual_gate_mode, "gauge": gauge,
                         "medrank": medrank, "gate": g,
                         "k1": a.r5dual_k1, "k2": a.r5dual_k2,
                         "ms": a.r5_ms, "ms_k": ms_k, "ms_rho": a.r5_ms_rho, "w_xc": a.r5_xc}
        print("  R5[dual] %s=%.4f -> gate=%.0f  (%.0fs)"
              % (a.r5dual_gate_mode, gauge, g, time.time() - t0), flush=True)
        for ck, cv in cand.items():
            fullsim[ck] = cv
            res["full_" + ck] = {"metrics": metrics(cv)}
            print("  R5[%-11s] -> %s" % (ck, fmt(metrics(cv))), flush=True)

    # R5': reference-anchored k-reciprocal re-ranking (opt-in, `--r5kr`).  A different
    # axis from `--csls`: a *set* operation over the reference image's mutual-neighbour
    # structure rather than a per-column scale.  It reads no query rows and no labels,
    # so the encoding is computed once, globally, and the same matrix is reused inside
    # the CV loop without touching the held-out rows.
    kz = None
    role_ch = None
    role_names = []
    if a.r5kr:
        probe = None
        if a.r5kr_anchor == "query":
            # the probe is the very representation the base score is built on, so the
            # channel answers "does this gallery item share neighbourhood structure
            # with *this* query" instead of "with some image near this query".
            probe = norm_rows((1.0 - a.r5kr_alpha) * T["d0"]
                              + a.r5kr_alpha * f_ref).astype(np.float32)
        elif a.r5kr_anchor == "role":
            # --- merged R5 (single module) -------------------------------------------
            # R3's role decomposition already supplies one query-side view per edit
            # role.  Instead of a single probe, every role view earns its own
            # query-conditioned reciprocal set against the gallery graph, and the
            # module keeps only the *role consensus*.  Two things follow, and both
            # are the point of merging the arms:
            #   * the probe is query-conditioned for every role, so the channel can
            #     never degenerate into the query-independent column prior that CSLS
            #     already models (the CIRCO failure of the reference-only probe);
            #   * agreement across roles is a label-free per-candidate statistic, so
            #     the neighbourhood term no longer rests on one global weight alone.
            role_names = [r.strip() for r in a.r5role_roles.split(",") if r.strip()]
            role_names = [r for r in role_names if r in TR]
            if not role_names:
                raise SystemExit("--r5kr_anchor role: no usable role views in %s"
                                 % a.r5role_roles)
            nq = TR[role_names[0]].shape[0]
            Pm = ((1.0 - a.r5kr_alpha) * np.concatenate(
                [TR[r] for r in role_names], 0)
                + a.r5kr_alpha * np.tile(f_ref, (len(role_names), 1)))
            probe = norm_rows(Pm).astype(np.float32)
            res["r5role"] = {"roles": role_names, "alpha": a.r5kr_alpha,
                             "stat": a.r5role_stat, "k1": a.r5kr, "k2": a.r5kr_k2}
            print("  R5[merged] role probes: %s (alpha=%.2f, k1=%d, %d probes)"
                  % (",".join(role_names), a.r5kr_alpha, a.r5kr, probe.shape[0]),
                  flush=True)
        kr = ref_kreciprocal(a.r5kr, a.r5kr_k2, a.r5kr_mix, probe=probe)
        if kr is not None:
            if a.r5kr_anchor == "role":
                nq = TR[role_names[0]].shape[0]
                role_ch = {r: kr[i * nq:(i + 1) * nq]
                           for i, r in enumerate(role_names)}
                St = np.stack([zrow(role_ch[r]).astype(np.float32)
                               for r in role_names])
                st = a.r5role_stat
                if st == "mean":
                    C = St.mean(0)
                elif st == "min":
                    C = St.min(0)                 # must be good under EVERY role
                elif st == "max":
                    C = St.max(0)                 # any role suffices
                elif st == "cnt":
                    C = (St > 0).sum(0).astype(np.float32)
                else:                             # gmean, shifted to stay positive
                    C = np.exp(np.log(np.maximum(St + 3.0, 1e-6)).mean(0))
                kz = zrow(C).astype(np.float32)
                res["r5role_alone"] = {r: metrics(role_ch[r]) for r in role_names}
                res["r5role_alone"]["cons"] = metrics(C)
                print("  R5[merged] %d role channels alone: %s" % (
                    len(role_names), "  ".join(
                        "%s=%.2f" % (r, metrics(role_ch[r])[main_key])
                        for r in role_names)), flush=True)
                print("  R5[merged] consensus(%s) alone -> %s"
                      % (st, fmt(metrics(C))), flush=True)
            else:
                kz = zrow(kr).astype(np.float32)
            # The query's own reference image is not a candidate answer, and its
            # k-reciprocal self-similarity is degenerate: Jaccard(V_ref, V_ref) = 1,
            # the maximum of the channel.  Left in, the channel would simply push the
            # reference image to rank 1.  Datasets that follow the CIRR/CIRCO protocol
            # mask that column anyway, so zeroing it is a no-op there; it is what makes
            # FIQ (whose official protocol does *not* mask) usable at all.
            kz[np.arange(kz.shape[0]), ref_idx] = 0.0
            # --- R5' service-region diagnostic (label-free, landed in the log).
            # `cmass_frac` = share of the channel's variance carried by the
            # query-independent gallery column mean  (what `--csls` already models).
            # `corr_col`   = correlation between that column profile and the column
            # profile of the fused score whose column mass CSLS removes.
            # `spearman_vs_base` = mean within-query Spearman redundancy between the
            # channel and the fused score.  High cmass/corr_col/spearman => after CSLS
            # nothing is left to add, which is exactly what `w*=0` reports.
            m_kr = kr.mean(0, keepdims=True)
            m_b = Bal.mean(0, keepdims=True)
            cmass = float(np.var(m_kr) / (np.var(kr) + 1e-12))
            rcol = float(np.corrcoef(m_kr.ravel(), m_b.ravel())[0, 1])
            rk = np.argsort(np.argsort(kz, axis=1), axis=1)
            rb = np.argsort(np.argsort(Bal, axis=1), axis=1)
            rk = rk - rk.mean(1, keepdims=True)
            rb = rb - rb.mean(1, keepdims=True)
            sp = float(np.mean((rk * rb).sum(1)
                               / (np.sqrt((rk ** 2).sum(1) * (rb ** 2).sum(1)) + 1e-12)))
            res["r5kr_diag"] = {"cmass_frac": cmass, "corr_col": rcol,
                                "spearman_vs_base": sp}
            print("  R5'[diag] cmass_frac=%.4f corr_col=%.4f "
                  "mean_spearman_vs_base=%.4f" % (cmass, rcol, sp), flush=True)
            k2v = a.r5kr_k2 or int(np.around(a.r5kr / 2.0))
            res["r5kr"] = {"k1": a.r5kr, "k2": k2v, "mix": a.r5kr_mix,
                           "alone": metrics(kr)}
            print("  R5'[k-reciprocal k1=%d k2=%d mix=%.2f] alone -> %s"
                  % (a.r5kr, k2v, a.r5kr_mix, fmt(metrics(kr))), flush=True)
            for tag in list(fullsim):
                s0 = fullsim[tag]
                sd0 = s0.std(1, keepdims=True)
                if a.r5kr_w is None:
                    cand = {w: metrics(s0 + w * sd0 * kz)[main_key] for w in KRGRID}
                    bw = max(cand, key=cand.get)
                else:
                    bw = float(a.r5kr_w)
                sK = (s0 + bw * sd0 * kz).astype(np.float32)
                fullsim[tag + "_kr"] = sK
                res["full_" + tag + "_kr"] = {"w": bw, "metrics": metrics(sK)}
                print("  +R5'[k-reciprocal] w=%.2f on RELIC_%-9s -> %s"
                      % (bw, tag, fmt(metrics(sK))), flush=True)

    # ------------------------------------------------------------ k-fold CV ----
    if not a.no_cv:
        fold = np.arange(n) % a.folds
        V3 = ("d0", "de", "rw")
        fams = ["M0_ab_only", "M1_3view", "M2a_calib", "M2b_refrep",
                "RELIC_role", "RELIC_all", "RELIC_all_ref", "RELIC_m2b"]
        if a.csls:
            fams.append("RELIC_csls")
            if a.r5kr:
                # the shipped configuration: R5 (CSLS) then R5' (k-reciprocal) on top
                fams.append("RELIC_csls_kr")
        if kz is not None:
            fams.append("RELIC_kr")
        held = {t: np.full((n, N), -1e9) for t in fams}
        per_fold = {t: [] for t in fams}
        print("\n=== %d-fold CV (selection metric = %s) ===" % (a.folds, main_key),
              flush=True)

        def tune_w3(tr):
            best = (None, -1.0)
            for w in simplex(0.1, len(V3)):
                sm = sum(wi * S[k] for wi, k in zip(w, V3))
                r = metrics(sm, tr)[main_key]
                if r > best[1]:
                    best = (w, r)
            return best[0]

        for f in range(a.folds):
            tr, te = fold != f, fold == f
            w3 = tune_w3(tr)
            sm1 = sum(wi * S[k] for wi, k in zip(w3, V3))
            held["M1_3view"][te] = sm1[te]
            per_fold["M1_3view"].append(metrics(sm1, te)[main_key])

            # M0: retune (alpha, beta) on d0 alone
            best = (-1.0, None)
            for al, be in ABGRID:
                s0 = score(T["d0"], al, be)
                r = metrics(s0, tr)[main_key]
                if r > best[0]:
                    best = (r, s0)
            Btr = best[1]
            held["M0_ab_only"][te] = Btr[te]
            per_fold["M0_ab_only"].append(metrics(Btr, te)[main_key])

            # M2a / M2b: recalibrated ab on the fused family, + ref repulsion
            bb = {t: (-1.0, None) for t in ("M2a_calib", "M2b_refrep")}
            for al, be in ABGRID:
                sc = {k: score(T[k], al, be) for k in V3}
                b = sum(wi * sc[k] for wi, k in zip(w3, V3))
                r = metrics(b, tr)[main_key]
                if r > bb["M2a_calib"][0]:
                    bb["M2a_calib"] = (r, b)
                for gr in G_REF:
                    if gr == 0.0:
                        continue
                    sg = b - gr * s_refimg
                    r2 = metrics(sg, tr)[main_key]
                    if r2 > bb["M2b_refrep"][0]:
                        bb["M2b_refrep"] = (r2, sg)
            for t in bb:
                held[t][te] = bb[t][1][te]
                per_fold[t].append(metrics(bb[t][1], te)[main_key])
            M2a = bb["M2a_calib"][1]

            # RELIC: channels from the d0 base, weights fitted on the train folds
            for tag, sets, ref in (("RELIC_role", ROLES, False),
                                   ("RELIC_all", VSETS, False),
                                   ("RELIC_all_ref", VSETS, True)):
                if not sets:
                    continue
                w, g, sF = ascend(Btr, sets, VALID, iters=3,
                                  gamma=(G_REF if ref else None))
                held[tag][te] = sF[te]
                per_fold[tag].append(metrics(sF, te)[main_key])
                if tag == "RELIC_all_ref":
                    wA, gA = w, g
                if tag == "RELIC_all" and a.csls:
                    # honest: r_g is estimated from the *train* folds only, so the
                    # held-out rows never contribute to their own calibration.
                    sC = csls_apply(sF, tr, a.csls, a.csls_lam)
                    held["RELIC_csls"][te] = sC[te]
                    per_fold["RELIC_csls"].append(metrics(sC, te)[main_key])
                    if a.r5kr:
                        # R5' on top of the shipped R5 model.  Same honest protocol:
                        # the encoding is a gallery-side index, so only the mixing
                        # weight is fitted, and only on the train folds.
                        sd_c = sC.std(1, keepdims=True)
                        gg = (KRGRID if a.r5kr_w is None else (float(a.r5kr_w),))
                        cand = {v: metrics(sC + v * sd_c * kz, tr)[main_key] for v in gg}
                        vbest = max(cand, key=cand.get)
                        sCK = (sC + vbest * sd_c * kz).astype(np.float32)
                        held["RELIC_csls_kr"][te] = sCK[te]
                        per_fold["RELIC_csls_kr"].append(metrics(sCK, te)[main_key])
                if tag == "RELIC_all" and kz is not None:
                    # R5': only the mixing weight is fitted, and on the train folds;
                    # the encoding itself is query-independent, so no held-out row
                    # contributes to it in any way.
                    sd_k = sF.std(1, keepdims=True)
                    gg = (KRGRID if a.r5kr_w is None else (float(a.r5kr_w),))
                    cand = {v: metrics(sF + v * sd_k * kz, tr)[main_key] for v in gg}
                    vbest = max(cand, key=cand.get)
                    sK = (sF + vbest * sd_k * kz).astype(np.float32)
                    held["RELIC_kr"][te] = sK[te]
                    per_fold["RELIC_kr"].append(metrics(sK, te)[main_key])
            # RELIC on top of the strongest single-vector base (M2a fused) + gamma
            sets = VSETS
            sd = M2a.std(1, keepdims=True)
            CHm = {k: calz(k) * sd * VALID[k][:, None] for k in sets}
            w2 = {k: wA.get(k, 0.0) for k in sets}
            cand = {}
            for gr in G_REF:
                sm = M2a + sum(w2[k] * CHm[k] for k in sets) - gr * s_refimg
                cand[gr] = (metrics(sm, tr)[main_key], sm)
            gbest = max(cand, key=lambda x: cand[x][0])
            sb = cand[gbest][1]
            held["RELIC_m2b"][te] = sb[te]
            per_fold["RELIC_m2b"].append(metrics(sb, te)[main_key])

            print("  fold %d: w3=%s %s" % (
                f, tuple(round(x, 2) for x in w3),
                {t: "%.2f" % per_fold[t][-1] for t in fams if per_fold[t]}), flush=True)

        print("\n=== CV summary (held-out) ===", flush=True)
        cv = {}
        for t in fams:
            if not per_fold[t]:      # channel set emptied by --drop (e.g. tgonly)
                print("  %-15s skipped (no channels)" % t, flush=True)
                continue
            m = float(np.mean(per_fold[t]))
            full = metrics(held[t])
            cv[t] = {"fold_mean": m, "fold_std": float(np.std(per_fold[t])), "full": full}
            res["CV_" + t] = full
            print("  %-15s fold_mean=%.3f +-%.3f  full=%s"
                  % (t, m, float(np.std(per_fold[t])), fmt(full)), flush=True)
        res["cv"] = cv
        if name.startswith("fiq"):
            res["CV_full_summary"] = {t: aggregate_fiq({name: cv[t]["full"]})
                                      for t in fams if t in cv}

    if a.dump:
        # D5/D7 artefacts: fp16 similarity matrices so that paired bootstrap tests
        # and the qualitative cases can be recomputed without touching the GPU.
        mats = {"base_ab0": S["d0"], "base_tuned": Bal}
        mats.update({"full_" + t: m for t, m in fullsim.items()})
        if not a.no_cv:
            mats.update({"cv_" + t: held[t] for t in fams})
        err = abs(perq(Bal).mean() - metrics(Bal)[main_key])
        print("  dump: |mean(perq_base) - %s| = %.4f" % (main_key, err), flush=True)
        arrs = {"perq_base_tuned": perq(Bal).astype(np.float32)}
        arrs.update({k: v.astype(np.float16) for k, v in mats.items()})
        if kz is not None:
            arrs["ch_kr"] = kz.astype(np.float16)
        for k, v in r5dual_mats.items():
            arrs["ch_" + k] = v.astype(np.float16)
        if role_ch:
            # raw per-role Jaccard channels of the merged R5, so any consensus
            # operator or label-free gate can be re-tested offline (no GPU run).
            for r in role_names:
                arrs["chrole_" + r] = role_ch[r].astype(np.float16)
        # the exact role channels that entered the score (variance matched), so a
        # qualitative case can be traced back to the evidence that moved the rank
        sd_full = Bal.std(1, keepdims=True)
        for k in VSETS:
            ch = calz(k) * sd_full * VALID[k][:, None]
            arrs["ch_" + k] = ch.astype(np.float16)
        np.savez_compressed(a.dump, **arrs)
        print("wrote", a.dump, flush=True)

    res["timing"] = timing
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
