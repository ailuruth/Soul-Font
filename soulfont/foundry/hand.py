"""What the writer's own Korean cells settle about the font.

Two things are read off the 33 style cells besides the component features the model
takes from them.

The frame. DM-Font was trained on fonts rendered one frame per font: every glyph centred
in a square the size of that font's largest glyph, so a small syllable stays small. The
sheet used to be cropped one frame per *cell*, trimming each mark and adding a fixed
30 px border, so every reference filled its own frame and how much of it depended on
how large the writer wrote. frame_sheet() gives the whole sheet one frame instead, sized
off its marks, the way handwriting-fonts frames a hand (one frame per font, on the 90th
percentile rather than the largest mark, so one oversized syllable cannot shrink the rest).

Measured with scripts/eval_generation.py on 14 OFL handwriting fonts, 200 unseen
syllables each, after fitting each font to one scale the way refine_metrics does:

                            writing ~110 px tall     writing ~55 px tall
                            IoU     SSIM             IoU     SSIM
    one frame per cell      0.467   0.634            0.453   0.618
    one frame per sheet     0.471   0.639            0.479   0.635

The sheet frame won on 11 of the 14 fonts at the larger size and 13 at the smaller. The
per-cell frame lost most where writing was small, which tall template cells invite; the
sheet frame makes the result independent of it.

The pen. The model draws unseen syllables about an eighth heavier than the hand it was
shown (ink 1.13x the writer's across the same 14 fonts), but reconstructs the style cells
themselves at the writer's weight. So the correction is measured where the bias lives:
the writer's references against a sample of generated syllables. Moving the generated
set to the writer's pen (mostly thinner; thicker for a heavy pen) raised SSIM on 12 of 14
fonts and left the other two unchanged (0.639 -> 0.644; 11 and 3 at the smaller size),
with IoU flat (0.471 -> 0.470) and ink down from 1.13x to 1.10x. It also keeps the writer's own 33 syllables, which go into the
font as written, the same weight as the 11,139 generated around them.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

# Same ink/paper split as FontStyleProcessor.clean_images.
PAPER_LEVEL = 200
# crop.py hands every cell over at this size; the frame is cut and squared at it too.
WORK_SIZE = 512
# The sheet frame is sized so a mark at this percentile of the sheet's marks fills this
# share of it. Swept on the 14 fonts, this sits on a plateau: the 75th percentile at 0.80,
# the median at 0.75 and the 90th at 0.80 all land within 0.003 IoU of it. Off the plateau
# it falls: the largest mark at 1.0 scored 0.456, and the 90th at 0.75 starts to slip.
FRAME_PERCENTILE = 90
FRAME_FILL = 0.85

# Pen width is read with every raster brought to WORK_SIZE: at the model's 128 px a stroke
# is 4-8 px, and a whole-pixel estimate cannot see a 10% difference.
# Never move an edge by more than this share of the writer's pen. The measured bias is
# about a tenth; anything far past that is more likely an estimator failure than the model.
MAX_PEN_CORRECTION = 0.15


def _ink_box(gray):
    """(left, top, right, bottom) of the ink in a greyscale array, or None if blank."""
    ink = gray <= PAPER_LEVEL
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    if not rows.size:
        return None
    return cols[0], rows[0], cols[-1] + 1, rows[-1] + 1


def frame_sheet(cells, size):
    """Square, binarized rasters of the marks in `cells`, all in one shared frame.

    `cells` are images cut from one sheet at one scale (crop.py's framed/ crops), so a
    pixel means the same distance in every one. Each mark is centred in a square whose side
    comes from the whole sheet; a mark larger than that keeps its own size. Returns
    `size` x `size` 'L' images, ink black on white.
    """
    grays = [np.asarray(c.convert('L')) for c in cells]
    boxes = [_ink_box(g) for g in grays]
    sides = [max(b[2] - b[0], b[3] - b[1]) for b in boxes if b]
    frame = int(round(np.percentile(sides, FRAME_PERCENTILE) / FRAME_FILL)) if sides else 1

    out = []
    for gray, box in zip(grays, boxes):
        if box is None:
            out.append(Image.new('L', (size, size), 255))
            continue
        mark = gray[box[1]:box[3], box[0]:box[2]]
        side = max(frame, *mark.shape)
        square = np.full((side, side), 255, dtype=np.uint8)
        top, left = (side - mark.shape[0]) // 2, (side - mark.shape[1]) // 2
        square[top:top + mark.shape[0], left:left + mark.shape[1]] = mark
        work = np.asarray(Image.fromarray(square).resize((WORK_SIZE, WORK_SIZE), Image.BICUBIC))
        img = Image.fromarray(np.where(work > PAPER_LEVEL, 255, 0).astype(np.uint8))
        out.append(img if size == WORK_SIZE else img.resize((size, size), Image.LANCZOS))
    return out


def pen_width(images):
    """Median stroke width of a set of glyph rasters, as a share of the frame's width.

    A share rather than pixels, so rasters at different resolutions compare directly.
    """
    from skimage.morphology import skeletonize

    widths = []
    for img in images:
        big = img.convert('L').resize((WORK_SIZE, WORK_SIZE), Image.BILINEAR)
        mask = np.asarray(big) < 128
        if mask.sum() < 64:
            continue
        spine = skeletonize(mask)
        if spine.sum() < 8:
            continue
        half = float(np.median(ndimage.distance_transform_edt(mask)[spine]))
        widths.append(2.0 * half / WORK_SIZE)
    return float(np.median(widths)) if widths else 0.0


def pen_correction(written, generated):
    """How far to move every edge of `generated` to match `written`, as a share of the
    frame's width (+ thickens). Both sets must be framed alike; resolution may differ."""
    want, have = pen_width(written), pen_width(generated)
    if want <= 0 or have <= 0:
        return 0.0
    limit = MAX_PEN_CORRECTION * want
    return float(np.clip((want - have) / 2.0, -limit, limit))
