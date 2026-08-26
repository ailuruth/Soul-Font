"""Check the exported ONNX graphs still produce the PyTorch model's glyphs.

The exported graphs are the model the desktop app ships, and nothing downstream would
notice if they drifted: a wrong graph still emits 11,172 plausible-looking glyphs and
still builds a valid TTF. Only comparing the two backends catches it, so CI runs this
between exporting and packaging.

Tolerance is set by what the pipeline already treats as the same font. foundry/inference
documents that MPS and CPU runs differ by up to 2.5e-4 per pixel and move a contour point
on about 1.5% of glyphs, and calls that identical output. The gate here is the stricter
of the two measures — how many pixels actually change side of the ink threshold, which is
all the vectorizer sees — held at 0.01%.

Exit code is 0 on pass, 1 on fail, so CI stops before it packages a broken model.

Usage:
    python scripts/verify_onnx.py [--samples 64] [--max-flip-pct 0.01]
"""
import argparse
import os
import sys

import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

MODEL_CFG = dict(C=32, n_comps=68, n_comp_types=3, language="kor")


def _style_images(chars):
    """Render the style set from a system font — this checks the graphs, not a template."""
    from PIL import Image, ImageDraw, ImageFont

    candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        os.path.join(PROJ_ROOT, "media", "ttf_files", "MaruBuri-Regular.ttf"),
        "C:\\Windows\\Fonts\\malgun.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        sys.exit("[verify] no Korean font available to render style references")

    font = ImageFont.truetype(path, 96)
    out = []
    for ch in chars:
        im = Image.new("L", (128, 128), 255)
        d = ImageDraw.Draw(im)
        box = d.textbbox((0, 0), ch, font=font)
        d.text(((128 - (box[2] - box[0])) / 2 - box[0],
                (128 - (box[3] - box[1])) / 2 - box[1]), ch, font=font, fill=0)
        out.append(((np.asarray(im, np.float32) / 255.0) - 0.5) / 0.5)
    return np.stack(out)[:, None, :, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default=os.path.join(PROJ_ROOT, "onnx"))
    ap.add_argument("--checkpoint",
                    default=os.path.join(PROJ_ROOT, "checkpoints", "korean-handwriting.pth"))
    ap.add_argument("--samples", type=int, default=64,
                    help="target syllables to compare, spread across the whole block")
    ap.add_argument("--max-flip-pct", type=float, default=0.01,
                    help="allowed share of pixels changing side of the ink threshold")
    args = ap.parse_args()

    import torch

    from foundry.char_layout import KOR_STYLE_CHARS
    from foundry.inference import get_device, load_model
    from foundry.onnx_backend import OnnxGenerator, _decompose

    style = _style_images(KOR_STYLE_CHARS)
    style_ids = np.zeros(len(KOR_STYLE_CHARS), np.int64)
    style_comp = np.array([_decompose(c) for c in KOR_STYLE_CHARS], np.int64)

    # Spread the sample over the whole Hangul block rather than taking the first N, so
    # rare finals and rare medials are included rather than skipped.
    step = 11172 // args.samples
    targets = [chr(0xAC00 + i * step) for i in range(args.samples)]
    trg_ids = np.zeros(len(targets), np.int64)
    trg_comp = np.array([_decompose(c) for c in targets], np.int64)

    gen = load_model(args.checkpoint, MODEL_CFG, get_device("cpu"))
    gen.eval()
    with torch.no_grad():
        gen.reset_dynamic_memory()
        gen.encode_write(torch.from_numpy(style_ids), torch.from_numpy(style_comp),
                         torch.from_numpy(style), reset_dynamic_memory=False)
        ref = gen.read_decode(torch.from_numpy(trg_ids), torch.from_numpy(trg_comp)).numpy()

    onnx_gen = OnnxGenerator(args.onnx_dir)
    onnx_gen.encode_write(style_ids, style_comp, style)
    got = onnx_gen.read_decode(trg_ids, trg_comp)

    diff = np.abs(ref - got)
    flip_pct = 100.0 * ((ref > 0) != (got > 0)).mean()
    worst = int(np.argmax([((ref[i] > 0) != (got[i] > 0)).mean() for i in range(len(targets))]))

    print(f"[verify] {len(targets)} glyphs, providers={onnx_gen.providers}")
    print(f"[verify] mean |diff| {diff.mean():.3e}   max {diff.max():.3e}")
    print(f"[verify] threshold flips {flip_pct:.5f}%  (limit {args.max_flip_pct}%), "
          f"worst glyph {targets[worst]!r}")

    if flip_pct > args.max_flip_pct:
        print("[verify] FAIL — the exported graphs do not reproduce the PyTorch model")
        return 1
    print("[verify] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
