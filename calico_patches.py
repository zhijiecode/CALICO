#!/usr/bin/env python3
"""Build the gallery patch-token cache (RELIC/CALICO R2/R4) for ONE backbone.

Generic replacement of probe_late_interaction.py's cache builder, which was
hard-wired to ViT-L-14.  Reads a ``cirbench/<dataset>.json`` manifest and
writes ``out/lateint/<dataset>_patch.<tag>.npz`` with the same three members
as the L/14 caches:

    patch    (N, P, D_shared) fp16   post-ln_post @ vis.proj patch tokens
    cls      (N, D_shared)    fp32   L2-normalised global image features
    meanpool (N, D_shared)    fp32   L2-normalised mean of patch tokens

Sharding
--------
  --shard I --num-shards S   encode the strided rows I, I+S, ... into a raw
                             binary shard under
                             out/lateint/_shards_<dataset>_<tag>/
  --merge                    stream every row (in the original manifest order)
                             into the final .npz and delete the shard dir.

Shards are raw .bin files (not .npz) so the 123k-image bigG cache (~81 GB)
can be merged without materialising it in RAM: the npy data section is
streamed straight into a ZIP_STORED zip (= the on-disk .npz layout).

Usage
  python encode_cirbench_patches.py --dataset cirr_val --tag cliplaion_b32 \
      --shard 0 --num-shards 4 --gpu 2 --img-bs 128
  python encode_cirbench_patches.py --dataset cirr_val --tag cliplaion_b32 --merge
"""
import argparse
import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("CALICO_ROOT", Path(__file__).resolve().parent)).resolve()
MAN_DIR = ROOT / "manifests"
MODELS = ROOT / "models"
OUTDIR = ROOT / "out" / "lateint"

# tag -> (open_clip model name, weights file, fp16 vision compute)
SPECS = {
    "cliplaion": ("ViT-L-14",
                  MODELS / "clip_laion" / "open_clip_pytorch_model.bin",
                  False),
    "cliplaion_b32": ("ViT-B-32",
                      MODELS / "clip_laion_b32" / "open_clip_pytorch_model.bin",
                      False),
    "cliplaion_g14": ("ViT-bigG-14",
                      MODELS / "clip_laion_g14" / "open_clip_pytorch_model.bin",
                      False),
}


def shard_dir(dataset, tag):
    return OUTDIR / ("_shards_%s_%s" % (dataset, tag))


def out_path(dataset, tag):
    return OUTDIR / ("%s_patch.%s.npz" % (dataset, tag))


def npy_header(shape, dtype):
    buf = io.BytesIO()
    np.lib.format.write_array_header_2_0(
        buf, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
              "fortran_order": False, "shape": tuple(shape)})
    return buf.getvalue()


class NpyStreamer:
    """Write one .npy member row-by-row into an open zip (ZIP_STORED)."""

    def __init__(self, zf, name, shape, dtype):
        self.zf = zf
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        zi = zipfile.ZipInfo(name + ".npy", date_time=(2026, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_STORED
        zi.external_attr = 0o600 << 16
        self.fp = zf.open(zi, mode="w", force_zip64=True)
        self.fp.write(npy_header(self.shape, self.dtype))
        self.rows_done = 0

    def write_rows(self, arr):
        arr = np.ascontiguousarray(arr, dtype=self.dtype)
        assert arr.shape[1:] == self.shape[1:], (arr.shape, self.shape)
        self.fp.write(arr.tobytes(order="C"))
        self.rows_done += arr.shape[0]

    def close(self):
        assert self.rows_done == self.shape[0], (self.rows_done, self.shape[0])
        self.fp.close()


def run_shard(a, model_name, weights, use_fp16):
    # When the launcher already pinned CUDA_VISIBLE_DEVICES (multi-shard
    # drivers), --gpu is an index *inside that mask* and we must not overwrite
    # the env (overwriting after torch import re-selects physical devices and
    # two shards then collide on one GPU).  Only set the mask when absent.
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch
    import open_clip
    from PIL import Image

    man = json.load(open(MAN_DIR / ("%s.json" % a.dataset)))
    paths = list(man["corpus_paths"])
    data_root = a.data_root
    if data_root:
        for i, p in enumerate(paths):
            if not os.path.exists(str(p)):
                for anc in ("CIRR", "CIRCO", "FashionIQ"):
                    k = str(p).find("/%s/" % anc)
                    if k >= 0:
                        paths[i] = str(Path(data_root) / str(p)[k + 1:])
                        break
    N = len(paths)
    rows = list(range(a.shard, N, a.num_shards))

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=str(weights), device="cuda",
        force_quick_gelu=False)
    model.eval().requires_grad_(False)
    if use_fp16:
        model.half()
    vis = model.visual

    sd = shard_dir(a.dataset, a.tag)
    sd.mkdir(parents=True, exist_ok=True)
    pbin = sd / ("patch.shard%02d.bin" % a.shard)
    cbin = sd / ("cls.shard%02d.bin" % a.shard)

    # Parallel JPEG decode/preprocess (shuffle=False keeps manifest order).
    import torch.utils.data as tud

    class _ImgDS(tud.Dataset):
        def __len__(self):
            return len(rows)
        def __getitem__(self, k):
            r = rows[k]
            try:
                return preprocess(Image.open(paths[r]).convert("RGB"))
            except Exception:
                return torch.zeros(3, 224, 224)

    dl = tud.DataLoader(_ImgDS(), batch_size=a.img_bs, num_workers=a.workers,
                        shuffle=False, pin_memory=True, persistent_workers=False)

    t0 = time.time()
    n_done = 0
    with open(pbin, "wb") as fp_p, open(cbin, "wb") as fp_c, torch.no_grad():
        for x in dl:
            x = x.cuda(non_blocking=True)
            if use_fp16:
                x = x.half()
            d = vis.forward_intermediates(x, indices=1, stop_early=True,
                                         output_fmt="NLC",
                                         output_extra_tokens=False)
            pt = d["image_intermediates"][-1]          # (B,P,Wv)
            pt = vis.ln_post(pt).float() @ vis.proj    # (B,P,D) shared space
            clsf = torch.nn.functional.normalize(
                d["image_features"].float(), dim=-1)   # (B,D)
            fp_p.write(pt.half().cpu().numpy().tobytes(order="C"))
            fp_c.write(clsf.cpu().numpy().astype(np.float32).tobytes(order="C"))
            n_done += x.shape[0]
            if (n_done // a.img_bs) % 20 < 1:
                el = time.time() - t0
                eta = el / max(n_done, 1) * (len(rows) - n_done)
                print("  shard%d %d/%d (%.0fs eta %.0fs)"
                      % (a.shard, n_done, len(rows), el, eta), flush=True)
    json.dump({"rows": rows, "N": N}, open(sd / ("meta.shard%02d.json" % a.shard), "w"))
    print("SHARD %d/%d done: %d rows in %.0fs -> %s"
          % (a.shard, a.num_shards, n_done, time.time() - t0, sd), flush=True)


def run_merge(a):
    sd = shard_dir(a.dataset, a.tag)
    metas = sorted(sd.glob("meta.shard*.json"))
    if len(metas) != a.num_shards:
        raise SystemExit("expected %d shard metas, found %d in %s"
                         % (a.num_shards, len(metas), sd))
    info = [json.load(open(m)) for m in metas]
    N = info[0]["N"]
    assert all(m["N"] == N for m in info), "shard N mismatch"
    counts = [len(m["rows"]) for m in info]
    assert sum(counts) == N, (sum(counts), N)
    # guard against merging an incomplete shard set with silent row loss
    assert sorted(int(r) for m in info for r in m["rows"]) == list(range(N)), \
        "shard rows do not cover 0..N-1 exactly once"
    # rows are strided; shard I holds rows [I, I+S, ...].
    S = a.num_shards

    fp_p = [open(sd / ("patch.shard%02d.bin" % i), "rb") for i in range(S)]
    fp_c = [open(sd / ("cls.shard%02d.bin" % i), "rb") for i in range(S)]

    # infer dims from the first bytes of shard 0: P and D from the L cache
    # convention -- derive via manifest-free probe on one image is unnecessary:
    # patch rows are (P*D) fp16, cls rows D fp32; read D from file sizes.
    cls_bytes = fp_c[0].seek(0, 2)
    D = cls_bytes // counts[0] // 4
    assert D * 4 * counts[0] == cls_bytes
    pat_bytes = fp_p[0].seek(0, 2)
    PD = pat_bytes // counts[0] // 2
    assert PD % D == 0, "patch row size not divisible by D"
    P = PD // D
    for f in fp_p + fp_c:
        f.seek(0)
    print("merge: N=%d P=%d D=%d from %d shards" % (N, P, D, S), flush=True)

    # shard I local row l  <->  global row l*S + I
    def read_rows(fps, counts_i, nbytes_row, l0, l1, dtype):
        out = []
        for i in range(S):
            lo = max(l0, 0)
            hi = min(l1, counts_i[i])
            if hi <= lo:
                continue
            fps[i].seek(lo * nbytes_row)
            buf = fps[i].read((hi - lo) * nbytes_row)
            arr = np.frombuffer(buf, dtype=dtype).reshape(hi - lo, -1)
            for l in range(lo, hi):
                out.append((l * S + i, arr[l - lo]))
        out.sort(key=lambda t: t[0])
        return np.stack([t[1] for t in out], 0)

    tmp = out_path(a.dataset, a.tag).with_suffix(".npz.tmp")
    mp_tmp = sd / "meanpool.tmp.bin"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    B = 256
    # zipfile (py3.9) allows only one open write member at a time, so members
    # are streamed one after another.  meanpool is spooled to a small temp bin
    # during the patch pass (N x D x 4 bytes, ~0.6 GB even on the 123k gallery).
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as zf:
        sp = NpyStreamer(zf, "patch", (N, P, D), np.float16)
        mpf = open(mp_tmp, "wb")
        n_blocks = (max(counts) + B - 1) // B
        for bi in range(n_blocks):
            l0, l1 = bi * B, (bi + 1) * B
            pb = read_rows(fp_p, counts, P * D * 2, l0, l1, np.float16)
            pb = pb.reshape(pb.shape[0], P, D)
            mp = pb.astype(np.float32).mean(1)
            mp = mp / np.maximum(np.linalg.norm(mp, axis=1, keepdims=True), 1e-12)
            sp.write_rows(pb)
            mpf.write(mp.astype(np.float32).tobytes(order="C"))
            if bi % 40 == 0:
                print("  merge patch rows<%d (%.0fs)" % (l1, time.time() - t0),
                      flush=True)
        sp.close()
        mpf.close()

        sc = NpyStreamer(zf, "cls", (N, D), np.float32)
        for bi in range(n_blocks):
            l0, l1 = bi * B, (bi + 1) * B
            cb = read_rows(fp_c, counts, D * 4, l0, l1, np.float32)
            sc.write_rows(cb)
        sc.close()

        sm = NpyStreamer(zf, "meanpool", (N, D), np.float32)
        with open(mp_tmp, "rb") as mf:
            for r in range(N):
                buf = mf.read(D * 4)
                sm.write_rows(np.frombuffer(buf, dtype=np.float32).reshape(1, D))
        sm.close()
    for f in fp_p + fp_c:
        f.close()
    os.replace(tmp, out_path(a.dataset, a.tag))
    shutil.rmtree(sd)
    print("MERGE done -> %s (%.0fs), shards deleted"
          % (out_path(a.dataset, a.tag), time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True, choices=sorted(SPECS))
    ap.add_argument("--data-root", default=os.environ.get("CALICO_DATA_ROOT"),
                    help="re-root manifest image paths under this directory")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--img-bs", type=int, default=64)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()

    if a.merge:
        run_merge(a)
        return
    model_name, weights, use_fp16 = SPECS[a.tag]
    assert Path(weights).exists(), "missing weights: %s" % weights
    run_shard(a, model_name, weights, use_fp16)


if __name__ == "__main__":
    main()
