#!/usr/bin/env python3
"""Build a FAISS gallery index for one CALICO backbone over a bundled manifest.

Encodes every corpus image with ``normalize(backbone.encode_images(img))`` and
writes

    indices/<dataset>/<backbone>/{index.faiss, identifiers.npy, metadata.json}

The row order of the index is the manifest ``corpus_ids`` order; ``calico_eval``
asserts that the two line up (row r <-> corpus_ids[r]) and pairs row r with the
query labels.

Sharded encoding
  --shard I --num-shards S   encode strided rows I, I+S, ... into shards/
  --merge                    concatenate the shards, restore manifest order,
                             build the flat inner-product index and save it

Image paths come from the manifest's ``corpus_paths``.  If they do not exist
locally (the manifests were built on the authors' machine), either edit them or
set ``CALICO_DATA_ROOT`` / ``--data-root``: the stored path is then re-rooted
under that directory from the dataset folder (CIRR / CIRCO / FashionIQ) down.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from backbones import available_backbones, make_backbone
from faiss_utils import save_faiss_index

ROOT = Path(os.environ.get("CALICO_ROOT", Path(__file__).resolve().parent)).resolve()
MAN_DIR = ROOT / "manifests"
IDX_DIR = ROOT / "indices"

_ANCHORS = ("CIRR", "CIRCO", "FashionIQ")


def resolve_image(p, data_root):
    """Re-root a manifest image path under ``data_root`` when it is missing."""
    p = str(p)
    if not data_root or os.path.exists(p):
        return p
    for a in _ANCHORS:
        k = p.find("/%s/" % a)
        if k >= 0:
            return str(Path(data_root) / p[k + 1:])
    return str(Path(data_root) / Path(p).name)


class CorpusDataset(Dataset):
    def __init__(self, ids, paths, data_root, preprocess):
        self.ids, self.paths, self.dr, self.pp = ids, paths, data_root, preprocess

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        try:
            img = Image.open(resolve_image(self.paths[i], self.dr)).convert("RGB")
        except Exception:
            img = None
        if img is None:
            return None, i
        return self.pp(img), i


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def load_manifest(dataset, manifest, data_root):
    path = Path(manifest) if manifest else (MAN_DIR / ("%s.json" % dataset))
    with open(path) as f:
        m = json.load(f)
    return list(m["corpus_ids"]), list(m["corpus_paths"])


def run_shard(args):
    ids, paths = load_manifest(args.dataset, args.manifest, args.data_root)
    idxs = list(range(len(ids)))[args.shard::args.num_shards]
    bb = make_backbone(args.backbone)
    ds = CorpusDataset([ids[i] for i in idxs], [paths[i] for i in idxs],
                       args.data_root, bb.preprocess)
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers,
                    collate_fn=collate, shuffle=False, pin_memory=True)

    embs, out_ids, nfail = [], [], 0
    with torch.no_grad(), tqdm(total=len(idxs), desc="shard %d/%d" % (args.shard, args.num_shards)) as pbar:
        for tens, local in dl:
            valid = [k for k in range(len(local)) if tens[k] is not None]
            nfail += len(local) - len(valid)
            if valid:
                b = torch.stack([tens[k] for k in valid]).to(bb.device, non_blocking=True)
                f = bb.encode_images(b).float().cpu().numpy().astype(np.float32)
                for k in valid:
                    embs.append(f[k])
                    out_ids.append(ds.ids[local[k]])
            pbar.update(len(local))

    shard_dir = Path(args.out_dir) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    out = shard_dir / ("shard_%02d.npz" % args.shard)
    np.savez(out, embeddings=np.array(embs, dtype=np.float32),
             identifiers=np.array(out_ids))
    print("SHARD %d/%d done: %d encoded, %d failed -> %s"
          % (args.shard, args.num_shards, len(out_ids), nfail, out))


def run_merge(args):
    shard_dir = Path(args.out_dir) / "shards"
    files = sorted(shard_dir.glob("shard_*.npz"))
    if not files:
        raise SystemExit("no shard files in %s" % shard_dir)
    embs, ids = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        embs.append(d["embeddings"])
        ids.extend(d["identifiers"].tolist())
        print("loaded %s: %d" % (f.name, len(d["identifiers"])))
    embs = np.concatenate(embs, axis=0).astype(np.float32)
    ids = np.array(ids)
    dim = embs.shape[1]
    print("merged %d vectors, dim=%d" % (len(ids), dim))

    # Strided shards concatenate in shard order, not manifest order; restore it.
    m_ids, _paths = load_manifest(args.dataset, args.manifest, args.data_root)
    rank = {k: i for i, k in enumerate(m_ids)}
    assert set(ids.tolist()) == set(m_ids), \
        "shard id set != manifest corpus (missing/duplicate images)"
    perm = np.argsort(np.array([rank[k] for k in ids.tolist()],
                               dtype=np.int64), kind="stable")
    embs = np.ascontiguousarray(embs[perm])
    ids = ids[perm]
    assert ids.tolist() == m_ids

    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    save_faiss_index(index, ids, Path(args.out_dir), {
        "backbone": args.backbone, "embedding_dim": dim,
        "num_images": int(len(ids)),
        "manifest": str(Path(args.manifest) if args.manifest else
                        (MAN_DIR / ("%s.json" % args.dataset))),
    })
    print("MERGE done: %d vectors -> %s" % (index.ntotal, args.out_dir))


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="manifest name, e.g. cirr_val")
    ap.add_argument("--backbone", default="cliplaion", choices=available_backbones())
    ap.add_argument("--manifest", default=None, help="override manifest JSON")
    ap.add_argument("--data-root", default=os.environ.get("CALICO_DATA_ROOT"),
                    help="re-root manifest image paths under this directory")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = str(IDX_DIR / args.dataset / args.backbone)
    return args


def main():
    args = parse_args(sys.argv[1:])
    if args.merge:
        run_merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
