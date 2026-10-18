#!/usr/bin/env python3
"""Backbone adapters for the Phase-1 protocol runner.

The retrieval pipeline is model-agnostic; only a few things differ per backbone.

Two feature "views" are exposed:

  * ``encode_images`` / ``encode_texts`` -> L2-normalised embeddings, used for
    the corpus index and for the single-modality query modes (image_only /
    text_only), matching each model's official encoder.
  * ``query_image_feats`` / ``query_text_feats`` -> whatever the model's own
    composed-query code expects, i.e. the *raw* projected features for bgevl
    and the normalised features for metaclip2.

Fusion semantics (must match each model's official code):
  * metaclip2 : normalize(alpha * text + (1 - alpha) * image)   [weighted sum]
                over normalised features
  * bgevl     : normalize(text_raw + image_raw)                 [late fusion]
                summed in *raw* space (see modeling_MMRet_CLIP.encode_multimodal)

Corpus encoding is identical for both: normalize(image_features).

Backbones loaded from a local ``.bin`` (cliplaion, openai_vitl14) cannot pick up
the sibling ``open_clip_config.json``, so open_clip silently falls back to the
architecture defaults.  For OpenAI CLIP that default is wrong (plain GELU
instead of quick_gelu, which shifts image embeddings to cos ~0.88), hence the
explicit ``force_quick_gelu`` class attribute below.
"""
import os
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get("CALICO_ROOT", Path(__file__).resolve().parent)).resolve()


class OpenClipBackbone:
    """Base for plain-CLIP backbones that share the official fusion recipe
    normalize(alpha * text + (1 - alpha) * image) over normalised features."""

    name = "openclip"
    model_name = None
    pretrained = None
    index_subdir = None
    # open_clip's own pretrained tag for OpenAI CLIP sets this; a local .bin
    # load would otherwise default to plain GELU.
    force_quick_gelu = False

    def __init__(self):
        import open_clip
        self.device = torch.device("cuda")
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device,
            force_quick_gelu=self.force_quick_gelu)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.index_dir = ROOT / "indices" / self.index_subdir

    def encode_images(self, batch):
        with torch.no_grad():
            return F.normalize(self.model.encode_image(batch), dim=-1)

    def encode_texts(self, texts):
        with torch.no_grad():
            toks = self.tokenizer(list(texts)).to(self.device)
            return F.normalize(self.model.encode_text(toks), dim=-1)

    # official composed query operates on normalised features
    def query_image_feats(self, batch):
        return self.encode_images(batch)

    def query_text_feats(self, texts):
        return self.encode_texts(texts)

    def fuse_images(self, f1, f2):
        """Multi-image reference: normalize((img1 + img2) / 2)."""
        return F.normalize((f1 + f2) / 2, dim=-1)

    def combine(self, img_emb, txt_emb, alpha=0.8):
        return F.normalize(alpha * txt_emb + (1 - alpha) * img_emb, dim=-1)


class MetaCLIP2Backbone(OpenClipBackbone):
    """The backbone PinPoint ships as its MetaCLIP2 baseline."""

    name = "metaclip2"
    model_name = "ViT-H-14-worldwide-quickgelu"
    pretrained = "metaclip2_worldwide"
    index_subdir = "metaclip2"


class CLIPLaionBackbone(OpenClipBackbone):
    """Independent third-party Web CLIP: LAION-2B ViT-L/14.

    Deliberately neither MetaCLIP2 nor BGE-VL: a different model family, a
    different training corpus (LAION-2B) and a different (non-LLM) fusion, used
    to test whether the ESIA ladder generalises off the backbones we tuned on.
    """

    name = "cliplaion"
    model_name = "ViT-L-14"
    pretrained = str(ROOT / "models" / "clip_laion" / "open_clip_pytorch_model.bin")
    index_subdir = "clip_laion"



class CLIPLaionB32Backbone(OpenClipBackbone):
    """LAION-2B CLIP ViT-B/32 (open_clip tag laion2b_s34b_b79k).

    Same LAION-2B family / training corpus as ``cliplaion`` (ViT-L/14); used
    for the three-backbone column (B/32, L/14, G/14) matching SDR-CIR Table 1.
    """

    name = "cliplaion_b32"
    model_name = "ViT-B-32"
    pretrained = str(ROOT / "models" / "clip_laion_b32" / "open_clip_pytorch_model.bin")
    index_subdir = "cliplaion_b32"


class CLIPLaionG14Backbone(OpenClipBackbone):
    """LAION-2B CLIP ViT-bigG/14 (open_clip tag laion2b_s39b_b160k)."""

    name = "cliplaion_g14"
    model_name = "ViT-bigG-14"
    pretrained = str(ROOT / "models" / "clip_laion_g14" / "open_clip_pytorch_model.bin")
    index_subdir = "cliplaion_g14"


class OpenAIViTL14Backbone(OpenClipBackbone):
    """Official OpenAI CLIP ViT-L/14.

    The backbone most zero-shot CIR papers (CIRR / FashionIQ leaderboards)
    report numbers with, so it is what our tables are comparable to.  Local
    copy of ``timm/vit_large_patch14_clip_224.openai`` -- exactly what
    open_clip's own ``ViT-L-14/openai`` tag resolves to.
    """

    name = "openai_vitl14"
    model_name = "ViT-L-14"
    pretrained = str(ROOT / "models" / "clip_openai" / "open_clip_pytorch_model.bin")
    index_subdir = "clip_openai"
    force_quick_gelu = True


class BGEVLBackbone:
    name = "bgevl"
    index_subdir = "bge_vl_large"

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        self.device = torch.device("cuda")
        p = str(ROOT / "models" / "bge_vl_large")
        # Model config carries torch_dtype=bfloat16; load in fp32 so the
        # corpus/query embeddings are float32 and no manual casting is needed.
        self.model = CLIPModel.from_pretrained(
            p, torch_dtype=torch.float32).to(self.device).eval()
        try:
            self.processor = CLIPProcessor.from_pretrained(p)
        except Exception:
            self.processor = CLIPProcessor.from_pretrained(p, use_fast=False)
        self.image_processor = self.processor.image_processor
        self.index_dir = ROOT / "indices" / self.index_subdir

    def preprocess(self, img):
        return self.image_processor(img, return_tensors="pt")["pixel_values"][0]

    def encode_images(self, batch):
        with torch.no_grad():
            return F.normalize(
                self.model.get_image_features(pixel_values=batch), dim=-1)

    def encode_texts(self, texts):
        with torch.no_grad():
            enc = self._text_inputs(texts)
            return F.normalize(self.model.get_text_features(**enc), dim=-1)

    def _text_inputs(self, texts):
        enc = self.processor(text=list(texts), return_tensors="pt",
                             padding=True, truncation=True, max_length=77)
        return {k: v.to(self.device) for k, v in enc.items()}

    # official composed query (encode_multimodal) sums *raw* projected features
    def query_image_feats(self, batch):
        with torch.no_grad():
            return self.model.get_image_features(pixel_values=batch)

    def query_text_feats(self, texts):
        with torch.no_grad():
            return self.model.get_text_features(**self._text_inputs(texts))

    def fuse_images(self, f1, f2):
        """Multi-image reference: average raw features (text is added later)."""
        return (f1 + f2) / 2

    def combine(self, img_emb, txt_emb, alpha=None):
        # encode_multimodal: normalize(text_features + image_features)
        return F.normalize(img_emb + txt_emb, dim=-1)


_REGISTRY = {"metaclip2": MetaCLIP2Backbone, "bgevl": BGEVLBackbone,
             "cliplaion": CLIPLaionBackbone,
             "openai_vitl14": OpenAIViTL14Backbone,
             "cliplaion_b32": CLIPLaionB32Backbone,
             "cliplaion_g14": CLIPLaionG14Backbone}


def available_backbones():
    return sorted(_REGISTRY)


def index_dir_of(name):
    """Where ``name``'s FAISS index lives (no model weights are loaded)."""
    if name not in _REGISTRY:
        raise ValueError("unknown backbone: %s" % name)
    return ROOT / "indices" / _REGISTRY[name].index_subdir


def make_backbone(name):
    if name not in _REGISTRY:
        raise ValueError("unknown backbone: %s" % name)
    return _REGISTRY[name]()
