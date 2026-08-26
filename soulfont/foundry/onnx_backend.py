"""ONNX Runtime backend for the DM-Font generator.

Same two-stage shape as the PyTorch model — encode style images into a component memory
bank, then decode target syllables out of it — with the two learned halves running as
ONNX graphs and the bank itself in numpy. See scripts/export_onnx.py for why the seam is
there.

This exists so a packaged desktop build does not have to carry PyTorch: torch is 312 MB
against onnxruntime's ~60 MB, and on a Windows machine with no NVIDIA card ONNX Runtime's
CPU kernels are considerably quicker than torch's. The graphs hold the same weights as
checkpoints/korean-handwriting.pth, so output is the same font.

The bank is a faithful copy of models/memory.DynamicMemory: a dict keyed by
(style, component address) holding a list of feature tensors, averaged on read. Writing
the same component twice is not a mistake — it is how the model combines several
observations of one component, and the template set relies on it.
"""
import json
import os

import numpy as np

# Hangul component counts, matching datasets/kor_decompose.
_N_CHO, _N_JUNG = 19, 21


def comp_id_to_addr(comp_ids):
    """[B, 3] component ids -> flat memory addresses.

    numpy copy of models.memory.comp_id_to_addr for Korean: the three component types
    share one address space, so 중성 and 종성 are shifted past the ones before them.
    """
    addrs = np.asarray(comp_ids, dtype=np.int64).copy()
    addrs[:, 1] += _N_CHO
    addrs[:, 2] += _N_CHO + _N_JUNG
    return addrs


class MemoryBank:
    """numpy re-implementation of models.memory.DynamicMemory."""

    def __init__(self):
        self._slots = {}

    def reset(self):
        self._slots = {}

    def write(self, style_ids, addrs, feats):
        """feats: [B, n_comp_types, ...] — one feature per component of each character."""
        for style_id, char_addrs, char_feats in zip(style_ids, addrs, feats):
            per_style = self._slots.setdefault(int(style_id), {})
            for addr, feat in zip(char_addrs, char_feats):
                per_style.setdefault(int(addr), []).append(feat)

    def read(self, style_ids, addrs):
        """-> [B, n_comp_types, ...], averaging whatever was written to each address."""
        out = []
        for style_id, char_addrs in zip(style_ids, addrs):
            per_style = self._slots[int(style_id)]
            char = []
            for addr in char_addrs:
                feats = per_style[int(addr)]
                char.append(feats[0] if len(feats) == 1 else np.mean(feats, axis=0))
            out.append(np.stack(char))
        return np.stack(out)


class OnnxGenerator:
    """Drop-in stand-in for MACore's inference surface.

    Mirrors reset_dynamic_memory / encode_write / read_decode so foundry.inference can
    drive either backend without knowing which it has.
    """

    def __init__(self, onnx_dir, providers=None):
        import onnxruntime as ort

        with open(os.path.join(onnx_dir, "model.json"), encoding="utf-8") as f:
            self.meta = json.load(f)

        if providers is None:
            available = ort.get_available_providers()
            # CUDA where there is a card, CPU everywhere else. CoreML is deliberately not
            # in this list: it silently falls back per-op and has produced different
            # output on this graph, and a font that differs by machine is not acceptable.
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in available] or ["CPUExecutionProvider"]
        self.providers = providers

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.encoder = ort.InferenceSession(os.path.join(onnx_dir, "encoder.onnx"),
                                            opts, providers=providers)
        self.decoder = ort.InferenceSession(os.path.join(onnx_dir, "decoder.onnx"),
                                            opts, providers=providers)

        self.memory = MemoryBank()
        self.skip_memory = MemoryBank()

    def reset_dynamic_memory(self):
        self.memory.reset()
        self.skip_memory.reset()

    def encode_write(self, style_ids, comp_ids, style_imgs, reset_dynamic_memory=True):
        """style_imgs: [B, 1, 128, 128] float32 in [-1, 1]."""
        if reset_dynamic_memory:
            self.reset_dynamic_memory()

        style_imgs = np.ascontiguousarray(style_imgs, dtype=np.float32)
        comp_feats, skip_feats = self.encoder.run(None, {"style_imgs": style_imgs})

        addrs = comp_id_to_addr(comp_ids)
        style_ids = np.asarray(style_ids).reshape(-1)
        self.memory.write(style_ids, addrs, comp_feats)
        self.skip_memory.write(style_ids, addrs, skip_feats)
        return comp_feats

    def read_decode(self, target_style_ids, target_comp_ids):
        """-> [B, 1, 128, 128] float32 in [-1, 1]."""
        addrs = comp_id_to_addr(target_comp_ids)
        style_ids = np.asarray(target_style_ids).reshape(-1)

        mem = self.memory.read(style_ids, addrs).astype(np.float32)
        skip = self.skip_memory.read(style_ids, addrs).astype(np.float32)
        glyph, = self.decoder.run(None, {
            "mem": np.ascontiguousarray(mem),
            "comp_addrs": np.ascontiguousarray(addrs, dtype=np.int64),
            "skip_feats": np.ascontiguousarray(skip),
        })
        return glyph


def onnx_dir_if_present(project_root):
    """The exported model directory, or None if this install still has only the .pth."""
    d = os.path.join(project_root, "onnx")
    needed = ("encoder.onnx", "decoder.onnx", "model.json")
    return d if all(os.path.isfile(os.path.join(d, f)) for f in needed) else None


# --- torch-free replacements for the two dataset classes ---------------------------
#
# datasets/style_image_dataset.py and datasets/nonpaired_dataset.py only decompose a
# character and load an image, but both subclass torch.utils.data.Dataset, and importing
# them would drag torch back into a build whose whole point is not having it.

def _decompose(char):
    """(초성, 중성, 종성) ids. Copy of datasets.kor_decompose.decompose without the import."""
    code = ord(char) - 0xAC00
    if not 0 <= code < 11172:
        raise ValueError(f"{char!r} is not a complete Hangul syllable")
    return (code // (28 * 21), (code // 28) % 21, code % 28)


def load_style_image(path):
    """PIL image -> [1, 128, 128] float32 in [-1, 1].

    Matches foundry.inference.get_transform(): Grayscale -> Resize((128,128)) ->
    ToTensor -> Normalize([0.5], [0.5]). torchvision resizes a PIL image with BILINEAR,
    so that is what is used here; in practice the cleaned cells arrive at 128x128 already
    and it is a no-op, but the pipeline is not the only caller.
    """
    from PIL import Image

    img = Image.open(path).convert("L")
    if img.size != (128, 128):
        img = img.resize((128, 128), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return ((arr - 0.5) / 0.5)[None, :, :]


def style_batch(style_imgs, style_chars, style_id=0):
    """-> (style_ids [B], comp_ids [B, 3], imgs [B, 1, 128, 128])."""
    if len(style_imgs) != len(style_chars):
        raise ValueError("style_imgs and style_chars must be the same length")
    imgs = np.stack([load_style_image(p) for p in style_imgs])
    comp_ids = np.array([_decompose(c) for c in style_chars], dtype=np.int64)
    style_ids = np.full(len(style_imgs), style_id, dtype=np.int64)
    return style_ids, comp_ids, imgs


def decode_batches(chars, batch_size=128, style_id=0):
    """Yield (style_ids, comp_ids) per batch, in the order `chars` was given."""
    for i in range(0, len(chars), batch_size):
        chunk = chars[i:i + batch_size]
        yield (np.full(len(chunk), style_id, dtype=np.int64),
               np.array([_decompose(c) for c in chunk], dtype=np.int64))
