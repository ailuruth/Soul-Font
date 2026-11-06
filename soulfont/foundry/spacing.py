"""Side bearings from the white a reader sees, not from each glyph's bounding box.

Bounding-box spacing puts the same distance between the outermost points of every pair,
so a syllable whose edge is one thin stroke, or a long ㅡ sticking out at the bottom, gets
as much air as a solid one and the line reads as uneven. Type designers space by eye
instead; HT Letterspacer (huertatipografica/HTLetterspacer) turned that into a rule, and
this is our own implementation of the idea:

  * in a horizontal band where the script's body lives, scan the outline line by line;
  * on each side, measure how far in from the glyph's outermost point the ink sits, capped
    at a depth (the eye does not see to the back of an open counter) and allowed to change
    by at most one step per step (it does not follow a narrow crevice either);
  * the mean of that is the white the glyph already brings; the bearing takes some of it
    back.

How much it takes back was fitted to what designers did. The Korean handwriting fonts on
Google Fonts were stripped of their spacing, respaced by rule with each font's own average
bearing, and compared with the bearings their designers chose (scripts/eval_spacing.py;
Hangul only where it is proportional -- Nanum Pen, Nanum Brush, Dongle, East Sea Dokdo,
Cute Font, Yeon Sung, Stylish, Jua):

    mean error from the designer, 1000-unit em      left     right
      Hangul, one bearing for every syllable        10.3u    22.8u
      Hangul, this                                   9.0u    16.8u    14 of 16 font sides
      Latin/digits/marks, one bearing per kind      13.7u    13.1u
      Latin/digits/marks, this                      11.5u    11.3u    23 of 28 font sides

The constants are two numbers per side, and fitting them on all fonts but one and scoring
the one left out gives nearly the same errors (Hangul 9.4u / 17.7u), so this is not the
fit memorising its own answers.

Designers compensate harder on the right of a syllable than the left -- the stub of ㅏ or
ㅐ reads as further away than it is -- and only partly for Latin, so the share differs.
"""
import numpy as np
from fontTools.pens.basePen import BasePen

# Scanline spacing, in font units at a 1000 em.
STEP = 5
# Depth as a share of the band height, and the share of measured white the bearing gives
# back. Hangul sits on a plateau from 6.5% to 10% depth; Latin from 15% to 40%.
HANGUL_DEPTH = 0.08
HANGUL_GIVE_BACK = (0.9, 1.4)      # (left, right)
LATIN_DEPTH = 0.25                 # of the x-height, as HT Letterspacer measures it
LATIN_GIVE_BACK = (0.36, 0.40)


class _Segments(BasePen):
    """Flatten an outline into line segments, a few per curve."""

    def __init__(self, glyphset):
        super().__init__(glyphset)
        self.segs = []
        self._cur = self._start = None

    def _moveTo(self, p):
        self._cur = self._start = p

    def _lineTo(self, p):
        self.segs.append((self._cur, p))
        self._cur = p

    def _qCurveToOne(self, p1, p2):
        p0 = np.array(self._cur, float)
        a, b = np.array(p1, float), np.array(p2, float)
        t = np.linspace(0.0, 1.0, 5)[:, None]
        pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * a + t * t * b
        self.segs.extend(zip(map(tuple, pts[:-1]), map(tuple, pts[1:])))
        self._cur = p2

    def _curveToOne(self, p1, p2, p3):
        p0, a, b, c = (np.array(p, float) for p in (self._cur, p1, p2, p3))
        t = np.linspace(0.0, 1.0, 7)[:, None]
        pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * a + 3 * (1 - t) * t * t * b + t ** 3 * c
        self.segs.extend(zip(map(tuple, pts[:-1]), map(tuple, pts[1:])))
        self._cur = p3

    def _closePath(self):
        if self._cur != self._start:
            self.segs.append((self._cur, self._start))


def _extents(glyphset, name, ys):
    """Leftmost and rightmost ink on each scanline (NaN where there is none), and the
    glyph's full horizontal extent. None for an empty glyph."""
    pen = _Segments(glyphset)
    glyphset[name].draw(pen)
    if not pen.segs:
        return None
    s = np.asarray(pen.segs, dtype=float)
    x0, y0, x1, y1 = s[:, 0, 0], s[:, 0, 1], s[:, 1, 0], s[:, 1, 1]
    lo, hi = np.minimum(y0, y1), np.maximum(y0, y1)
    y = ys[:, None]
    hit = (lo <= y) & (hi > y)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    left = np.where(hit, x, np.inf).min(axis=1)
    right = np.where(hit, x, -np.inf).max(axis=1)
    left[np.isinf(left)] = np.nan
    right[np.isinf(right)] = np.nan
    return left, right, min(x0.min(), x1.min()), max(x0.max(), x1.max())


def _white(indent, depth, step):
    """Mean visible white on one side: the indent per scanline, capped and slope-limited."""
    d = np.clip(np.where(np.isnan(indent), depth, indent), 0.0, depth)
    for i in range(1, len(d)):
        d[i] = min(d[i], d[i - 1] + step)
    for i in range(len(d) - 2, -1, -1):
        d[i] = min(d[i], d[i + 1] + step)
    return float(d.mean())


def white(glyphset, name, band, depth, step=STEP):
    """(left, right) white a glyph brings to its own spacing, measured from its full extent.

    `band` is (bottom, top) in font units. Ink outside the band still sets the extremes, so
    a stroke that sticks out above or below the body counts as air beside the body -- which
    is what it looks like.
    """
    ys = np.arange(band[0], band[1] + step / 2.0, step)
    found = _extents(glyphset, name, ys)
    if found is None:
        return None
    left, right, xmin, xmax = found
    if np.all(np.isnan(left)):
        return depth, depth
    return (_white(left - xmin, depth, step), _white(xmax - right, depth, step))


def bearings(whites, mean_bearing, give_back):
    """Per-glyph (lsb, rsb): a glyph with more white of its own gets proportionally less.

    whites: {name: (white_left, white_right)}. The bearings average `mean_bearing`
    ((left, right), font units) over the set, so the overall density of the text is set by
    the caller and only its evenness comes from here.
    """
    if not whites:
        return {}
    w = np.array(list(whites.values()), dtype=float)
    k = np.asarray(give_back, dtype=float)
    out = np.asarray(mean_bearing, dtype=float) - k * (w - w.mean(axis=0))
    return {name: (float(l), float(r)) for name, (l, r) in zip(whites, out)}
