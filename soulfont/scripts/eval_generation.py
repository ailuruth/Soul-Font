"""Score how faithfully the generator continues a hand, against fonts whose glyphs are known.

A filled-in sheet has no answer key: nobody wrote the other 11,139 syllables. A handwriting
font does. Each font here plays a writer: its 33 template syllables are rendered the way a
pen leaves them on the sheet, pushed through the same reference preparation a real upload
gets, and the model's guesses at 200 syllables it never saw are compared with the font's
own. The comparison runs after one scale fit per font, the way refine_metrics fits the
Hangul, so glyph size inside the raster -- which the font build throws away -- does not
score.

    python scripts/eval_generation.py --fetch          # downloads 14 OFL fonts, once
    python scripts/eval_generation.py --write-px 55    # a writer with small handwriting

Prints, per font and on average, for each way of preparing the references:

    IoU    overlap of the ink after the fit (shape and weight together)
    SSIM   structural similarity after the fit
    ink    generated ink over the font's own ink (1.0 = the writer's weight)

Needs the exported ONNX model (scripts/export_onnx.py).
"""
import argparse
import glob
import os
import random
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps
from scipy import ndimage
from skimage.metrics import structural_similarity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from apppaths import DATA_ROOT  # noqa: E402
from foundry.char_layout import KOR_STYLE_CHARS  # noqa: E402
from foundry.font_processor import repen  # noqa: E402
from foundry.hand import frame_sheet, pen_correction  # noqa: E402
from foundry.onnx_backend import (OnnxGenerator, decode_batches, onnx_dir_if_present,  # noqa: E402
                                  style_batch)

# Korean handwriting families on Google Fonts, all SIL Open Font License.
OFL_FONTS = [
    "cutefont/CuteFont", "dokdo/Dokdo", "dongle/Dongle", "eastseadokdo/EastSeaDokdo",
    "gaegu/Gaegu", "gamjaflower/GamjaFlower", "himelody/HiMelody", "jua/Jua",
    "nanumbrushscript/NanumBrushScript", "nanumpenscript/NanumPenScript",
    "poorstory/PoorStory", "singleday/SingleDay", "stylish/Stylish", "yeonsung/YeonSung",
]
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/{}-Regular.ttf"
RENDER = 256


def fetch(dest):
    os.makedirs(dest, exist_ok=True)
    for name in OFL_FONTS:
        path = os.path.join(dest, os.path.basename(name) + "-Regular.ttf")
        if not os.path.exists(path):
            print(f"[eval] fetching {name}")
            urllib.request.urlretrieve(FONT_URL.format(name), path)


def _tight(font, ch):
    canvas = Image.new("L", (RENDER * 3, RENDER * 3), 255)
    ImageDraw.Draw(canvas).text((RENDER, RENDER), ch, font=font, fill=0)
    box = ImageChops.difference(canvas, Image.new("L", canvas.size, 255)).getbbox()
    return canvas.crop(box) if box else None


def truth_glyphs(path, chars, charset):
    """The font as DM-Font's training data was rendered: each glyph centred in a square
    the size of the font's largest glyph."""
    font = ImageFont.truetype(path, RENDER)
    pool = charset + [chr(c) for c in range(0x21, 0x7F)] + [chr(c) for c in range(0x3131, 0x3164)]
    side = max(max(g.size) for g in (_tight(font, c) for c in pool) if g is not None)
    out = []
    for ch in chars:
        g = _tight(font, ch)
        canvas = Image.new("L", (side, side), 255)
        canvas.paste(g, ((side - g.width) // 2, (side - g.height) // 2))
        out.append(np.asarray(canvas.resize((128, 128), Image.BILINEAR), dtype=np.float32) / 255)
    return np.stack(out)


def written_cells(path, write_px):
    """The 33 style syllables as a pen leaves them in their cells at 300 dpi: true relative
    sizes, a tall syllable about write_px high."""
    probe = ImageFont.truetype(path, RENDER)
    heights = sorted(_tight(probe, c).height for c in KOR_STYLE_CHARS)
    font = ImageFont.truetype(path, max(8, int(RENDER * write_px / heights[int(0.9 * (len(heights) - 1))])))
    cell = int(write_px * 1.6)
    cells = []
    for ch in KOR_STYLE_CHARS:
        g = _tight(font, ch)
        canvas = Image.new("L", (cell, cell), 255)
        canvas.paste(g, ((cell - g.width) // 2, (cell - g.height) // 2))
        cells.append(canvas)
    return cells


def per_cell_references(cells):
    """How references were prepared before one frame per sheet: each mark trimmed, given a
    30 px border at scan resolution, squared and filled to its own frame."""
    out = []
    for cell in cells:
        box = ImageChops.difference(cell, Image.new("L", cell.size, 255)).getbbox()
        mark = ImageOps.expand(cell.crop(box) if box else cell, border=30, fill=255)
        side = max(mark.size)
        square = Image.new("L", (side, side), 255)
        square.paste(mark, ((side - mark.width) // 2, (side - mark.height) // 2))
        a = np.asarray(square.resize((512, 512), Image.BICUBIC))
        out.append(Image.fromarray(np.where(a > 200, 255, 0).astype(np.uint8)).resize((128, 128), Image.LANCZOS))
    return out


def generate(gen, refs, targets):
    paths = []
    tmp = os.path.join(DATA_ROOT, "workdir", "eval_refs")
    os.makedirs(tmp, exist_ok=True)
    for i, img in enumerate(refs):
        paths.append(os.path.join(tmp, f"{i:02d}.png"))
        img.save(paths[-1])
    gen.encode_write(*style_batch(paths, KOR_STYLE_CHARS))
    outs = [gen.read_decode(t, c) for t, c in decode_batches(targets)]
    return [Image.fromarray(np.uint8(np.clip(o[0] * 0.5 + 0.5, 0, 1) * 255)) for o in np.concatenate(outs)]


def _height(a):
    rows = np.flatnonzero((a < 0.5).any(1))
    return rows[-1] - rows[0] + 1 if rows.size else 0


def score(pred, truth):
    """Mean IoU, SSIM and ink ratio after one zoom fits the whole set to the truth."""
    pred = np.stack([np.asarray(p.convert("L").resize((128, 128), Image.BOX), dtype=np.float32) / 255
                     for p in pred])
    zoom = np.median([_height(t) for t in truth]) / max(1, np.median([_height(p) for p in pred]))
    c = np.array([63.5, 63.5])
    rows = []
    for p, t in zip(pred, truth):
        f = ndimage.affine_transform(p, np.eye(2) / zoom, offset=c - c / zoom, order=1, cval=1.0)
        mf, mt = f < 0.5, t < 0.5
        rows.append(((mf & mt).sum() / max(1, (mf | mt).sum()),
                     structural_similarity(f, t, data_range=1.0),
                     mf.sum() / max(1, mt.sum())))
    return np.mean(rows, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonts", default=os.path.join(DATA_ROOT, "workdir", "eval_fonts"))
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--n", type=int, default=200, help="unseen syllables per font")
    ap.add_argument("--write-px", type=int, default=110, help="height of the writing on the sheet")
    args = ap.parse_args()

    if args.fetch:
        fetch(args.fonts)
    fonts = sorted(glob.glob(os.path.join(args.fonts, "*.ttf")))
    if not fonts:
        sys.exit(f"No fonts in {args.fonts}; run with --fetch.")
    onnx_dir = onnx_dir_if_present(PROJECT_ROOT)
    if not onnx_dir:
        sys.exit("No exported model; run scripts/export_onnx.py first.")

    with open(os.path.join(PROJECT_ROOT, "data", "charset", "korean2350.txt"), encoding="utf-8") as f:
        charset = [line.strip() for line in f if line.strip()]
    targets = random.Random(0).sample([c for c in charset if c not in KOR_STYLE_CHARS], args.n)
    gen = OnnxGenerator(onnx_dir, providers=["CPUExecutionProvider"])

    methods = ("per-cell frame", "sheet frame", "sheet frame + pen")
    table = {m: [] for m in methods}
    for path in fonts:
        truth = truth_glyphs(path, targets, charset)
        cells = written_cells(path, args.write_px)
        # crop.py's framed/ output: every cell squared and brought to 512 at one scale.
        sheet = frame_sheet([c.resize((512, 512), Image.BICUBIC) for c in cells], 128)
        generated = generate(gen, sheet, targets)
        delta = pen_correction(sheet, generated)
        runs = {
            "per-cell frame": generate(gen, per_cell_references(cells), targets),
            "sheet frame": generated,
            "sheet frame + pen": [repen(g, delta) for g in generated],
        }
        line = [f"{os.path.basename(path).split('-')[0]:17s}"]
        for m in methods:
            table[m].append(score(runs[m], truth))
            line.append("{} IoU {:.3f} SSIM {:.3f} ink {:.2f}".format(m, *table[m][-1]))
        print(" | ".join(line), flush=True)

    print(f"\nmean over {len(fonts)} fonts, {args.n} unseen syllables each, writing {args.write_px} px:")
    for m in methods:
        print("  {:18s} IoU {:.4f}  SSIM {:.4f}  ink {:.3f}".format(m, *np.mean(table[m], axis=0)))


if __name__ == "__main__":
    main()
