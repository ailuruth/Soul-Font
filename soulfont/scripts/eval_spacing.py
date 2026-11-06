"""Score foundry.spacing against the spacing designers gave their own fonts.

Each font has its side bearings thrown away and recomputed two ways -- one bearing for
every glyph of a kind (how Soul Font spaced Hangul before; its Latin had one bearing for
every kind, further off still) and foundry.spacing's optical rule --
each with the font's own average, so only how the bearing is *shared out* is compared. The
error is the mean distance from the designer's bearing, in units of a 1000 em. Hangul is
scored only on fonts whose Hangul is proportional; a fixed-width font has nothing to say
about per-syllable bearings.

    python scripts/eval_spacing.py                 # fonts from eval_generation.py --fetch
    python scripts/eval_spacing.py --fonts DIR
"""
import argparse
import glob
import os
import sys

import numpy as np
from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from apppaths import DATA_ROOT  # noqa: E402
from foundry import spacing  # noqa: E402
from foundry.refine_metrics import SPACE_MIN_PER_CAP, SPACE_PER_HANGUL_HEIGHT  # noqa: E402

KINDS = {
    'lower': [chr(c) for c in range(ord('a'), ord('z') + 1)],
    'upper': [chr(c) for c in range(ord('A'), ord('Z') + 1)],
    'digit': [chr(c) for c in range(ord('0'), ord('9') + 1)],
    'mark': list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"),
}


def _bounds(gs, name):
    pen = BoundsPen(gs)
    gs[name].draw(pen)
    return pen.bounds


def _score(gs, names, band, depth, give_back, upem):
    """(designer, box, optical) bearings for `names`, as [n, 2] arrays in 1/1000 em."""
    designer, whites = [], {}
    for n in names:
        b = _bounds(gs, n)
        w = spacing.white(gs, n, band, depth, step=spacing.STEP * upem / 1000)
        if b is None or w is None:
            continue
        designer.append((b[0], gs[n].width - b[2]))
        whites[n] = w
    designer = np.array(designer, dtype=float)
    mean = designer.mean(axis=0)
    optical = np.array(list(spacing.bearings(whites, mean, give_back).values()))
    box = np.broadcast_to(mean, designer.shape)
    scale = 1000.0 / upem
    return designer * scale, box * scale, optical * scale


def score_font(path, sample=400):
    f = TTFont(path)
    upem = f['head'].unitsPerEm
    gs, cmap = f.getGlyphSet(), f.getBestCmap()
    out = {}

    hangul = [cmap[c] for c in range(0xAC00, 0xD7A4) if c in cmap]
    hangul = hangul[::max(1, len(hangul) // sample)]
    advances = np.array([gs[n].width for n in hangul]) / upem
    bounds = [b for b in (_bounds(gs, n) for n in hangul) if b]
    if hangul and advances.std() > 0.02:
        band = (np.median([b[1] for b in bounds]), np.median([b[3] for b in bounds]))
        depth = spacing.HANGUL_DEPTH * (band[1] - band[0])
        out['hangul'] = _score(gs, hangul, band, depth, spacing.HANGUL_GIVE_BACK, upem)

    x_line, cap = _bounds(gs, cmap[ord('x')])[3], _bounds(gs, cmap[ord('H')])[3]
    parts = []
    for kind, chars in KINDS.items():
        names = [cmap[ord(c)] for c in chars if ord(c) in cmap]
        band = (0, x_line if kind == 'lower' else cap)
        parts.append(_score(gs, names, band, spacing.LATIN_DEPTH * x_line,
                            spacing.LATIN_GIVE_BACK, upem))
    out['latin'] = tuple(np.concatenate(p) for p in zip(*parts))

    if bounds and 32 in cmap:
        height = np.median([b[3] - b[1] for b in bounds])
        predicted = max(SPACE_MIN_PER_CAP * cap, SPACE_PER_HANGUL_HEIGHT * height)
        out['space'] = (gs[cmap[32]].width, predicted)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fonts', default=os.path.join(DATA_ROOT, 'workdir', 'eval_fonts'))
    args = ap.parse_args()
    fonts = sorted(glob.glob(os.path.join(args.fonts, '*.ttf')))
    if not fonts:
        sys.exit(f'No fonts in {args.fonts}; run scripts/eval_generation.py --fetch first.')

    totals = {'hangul': [], 'latin': []}
    spaces = []
    for path in fonts:
        result = score_font(path)
        line = [f'{os.path.basename(path).split("-")[0]:17s}']
        for script in ('hangul', 'latin'):
            if script not in result:
                line.append(f'{script} fixed-width')
                continue
            designer, box, optical = result[script]
            e_box = np.abs(designer - box).mean(axis=0)
            e_opt = np.abs(designer - optical).mean(axis=0)
            totals[script].append((e_box, e_opt))
            line.append(f'{script} L {e_box[0]:5.1f} -> {e_opt[0]:5.1f}u  R {e_box[1]:5.1f} -> {e_opt[1]:5.1f}u')
        if 'space' in result:
            actual, predicted = result['space']
            spaces.append(abs(predicted - actual) / actual)
            line.append(f'space {actual:.0f} vs {predicted:.0f}')
        print(' | '.join(line))

    print('\nmean error from the designer, one bearing per kind -> optical:')
    for script, rows in totals.items():
        if not rows:
            continue
        box = np.mean([r[0] for r in rows], axis=0)
        opt = np.mean([r[1] for r in rows], axis=0)
        wins = sum(int((r[1] < r[0]).sum()) for r in rows)
        print(f'  {script:6s} left {box[0]:5.1f} -> {opt[0]:5.1f}u   right {box[1]:5.1f} -> '
              f'{opt[1]:5.1f}u   ({wins} of {2 * len(rows)} font sides better)')
    if spaces:
        print(f'  word space: {100 * np.mean(spaces):.0f}% mean error from the designer')


if __name__ == '__main__':
    main()
