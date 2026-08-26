"""Central definition of the handwriting-template cell layout.

The template is cropped cell-by-cell in row-major order (see foundry/crop.py), producing
one cleaned 128x128 image per cell. This module fixes the *order* of those cells so the
pipeline knows which image is which character.

Layout contract (the physical template must follow this order):

    cells[0 : 33]                -> KOR_STYLE_CHARS   (Korean style references -> model)
    cells[33 : 33+len(EXTRA)]    -> EXTRA_CHARS       (English/digits/special -> traced
                                                       directly into the font, no model)

If a template only has the Korean cells, EXTRA_CHARS simply maps to nothing and the
pipeline behaves exactly as before.
"""
import math

TEMPLATE_LAYOUT_VERSION = "kor-33-plus-ascii-v5-10x13"

# The printed grid. 10x13 on A4 gives a cell of about 18.1 x 20.6 mm, against the 25.8 x
# 40.8 mm of the 4x7 sheet it replaces — roughly a third of the area, and close to square
# rather than a tall letterbox, which is the shape a Hangul syllable actually wants.
#
# The whole set now lands on a single page. That is the point of the smaller cell: 125
# guides at 4x7 took four sheets to print, fill and scan, and every extra sheet is another
# chance for one to be missed, fed in crooked, or scanned at a different exposure.
TEMPLATE_ROWS = 13
TEMPLATE_COLS = 10
TEMPLATE_CELLS_PER_PAGE = TEMPLATE_ROWS * TEMPLATE_COLS

# How much of each cell crop.py throws away, as a fraction of the cell.
#
# These live here rather than in crop.py because they are a contract between two files
# that never import each other: the cropper trims this much, and the template generator
# has to keep the printed guide glyph inside the trimmed band or the guide ends up traced
# into the font as though the user had drawn it. With the band expressed here, the
# generator can check its own artwork against the number the cropper will actually use
# (see scripts/generate_template.py, which refuses to write a sheet that would leak).
CELL_PAD_FRAC = 0.06     # trimmed off every cell edge — drops the printed grid line
TOP_EXTRA_FRAC = 0.14    # trimmed off the cell top as well — drops the printed guide glyph
# The band the guide has to fit inside, measured from the very top of the cell.
GUIDE_BAND_FRAC = CELL_PAD_FRAC + TOP_EXTRA_FRAC

# The Korean style references, in template-cell order.
#
# These 33 cells are the only ones the model sees. Between them they have to carry every
# Hangul component — all 19 초성, all 21 중성, all 28 종성 including "none" — because
# DM-Font composes the other 11,139 syllables out of the pieces it reads here. A component
# that never appears on this sheet is one the model has to guess at for the whole font.
#
# The old set met that requirement by walking the three component tables diagonally, which
# is the cheapest way to cover them — "각깪냓댼...", 28 cells, perfectly packed, every
# 종성 used exactly once. It also put 21 of the 28 outside KS X 1001 altogether, which
# is the standard's way of saying no Korean writes them, and that is what writers
# reported back: you cannot produce natural handwriting for a syllable you have never
# written in your life. Every one of the 33 below is inside KS X 1001.
#
# So the set is now chosen rather than constructed. It is a minimum-cost cover over the
# 2,350 syllables of KS X 1001 — the national standard's "commonly used" set — scored by
# how familiar each syllable is and constrained to the same total component coverage.
# Real syllables pack less tightly than a diagonal walk: the 겹받침 (ㄳ ㄵ ㄶ ㄺ ㄻ ㄼ ㄽ ㄾ
# ㄿ ㅀ ㅄ) occur in only a handful of real words each, and the rare medials ㅒ ㅖ ㅞ ㅢ
# crowd into very few 종성. That costs five extra cells — 28 -> 33 — which the sheet already
# had spare (125 guides on a 130-cell page). The page is unchanged and now exactly full.
#
# 31 of the 33 are ordinary words. 2 are not, and cannot be, because the language
# has nothing better to offer:
#   읊  the only syllable in KS X 1001 carrying the final ㄿ (읊다)
#   곬  the least obscure of the three carrying ㄽ (외곬); the others are 돐 and 옰
# Every other 종성 in Hangul has at least one everyday word behind it, and it is used here.
#
# KOR_STYLE_WORDS gives the word each syllable is recognisable from, so the writer can be
# shown what they are being asked to write. Both lists are ordered by 종성, which groups
# the similar-looking finals together on the sheet.
KOR_STYLE_WORDS = {
    "얘": "얘기",       "예": "예의",       "돼": "돼지",       "의": "의사",
    "퀵": "퀵서비스",     "밖": "바깥",       "넋": "넋",        "왼": "왼쪽",
    "앉": "앉다",       "찮": "괜찮다",      "뜯": "뜯다",       "쌀": "쌀밥",
    "읽": "읽다",       "굶": "굶다",       "짧": "짧다",       "곬": "외곬",
    "훑": "훑다",       "읊": "읊다",       "끓": "끓다",       "탐": "탐구",
    "웹": "웹사이트",     "값": "물건값",      "셋": "셋째",       "줬": "줬다",
    "빵": "빵집",       "왕": "왕자",       "룡": "공룡",       "맺": "맺다",
    "윷": "윷놀이",      "엌": "부엌",       "팥": "팥죽",       "옆": "옆집",
    "얗": "하얗다",
}
KOR_STYLE_CHARS = list("얘예돼의퀵밖넋왼앉찮뜯쌀읽굶짧곬훑읊끓탐웹값셋줬빵왕룡맺윷엌팥옆얗")
assert list(KOR_STYLE_WORDS) == KOR_STYLE_CHARS, "words and chars must stay in step"


def _assert_component_coverage():
    """Fail loudly if an edit to KOR_STYLE_CHARS drops a Hangul component.

    Silent partial coverage is the expensive failure here: the build still succeeds and
    still emits 11,172 syllables, but every syllable using the missing component is
    guesswork, and nobody finds out until they look at the font.
    """
    cho, jung, jong = set(), set(), set()
    for ch in KOR_STYLE_CHARS:
        code = ord(ch) - 0xAC00
        if not 0 <= code < 11172:
            raise ValueError(f"{ch!r} is not a complete Hangul syllable")
        cho.add(code // (28 * 21))
        jung.add((code // 28) % 21)
        jong.add(code % 28)
    for name, got, want in (("초성", cho, 19), ("중성", jung, 21), ("종성", jong, 28)):
        if len(got) != want:
            missing = sorted(set(range(want)) - got)
            raise ValueError(
                f"KOR_STYLE_CHARS covers only {len(got)}/{want} {name} "
                f"(missing indices {missing}). Every component must appear at least once "
                f"or the model has to invent it for the whole font."
            )


_assert_component_coverage()

# English uppercase, lowercase, digits.
ENG_CHARS = (
    [chr(c) for c in range(ord('A'), ord('Z') + 1)] +
    [chr(c) for c in range(ord('a'), ord('z') + 1)] +
    [chr(c) for c in range(ord('0'), ord('9') + 1)]
)

# Punctuation and symbols.
#
# The whole of printable ASCII, rather than a hand-picked twenty. Completing it is the
# boundary worth drawing: what fell outside the old set was $ < > [ \ ] ^ _ ` { | } ~ —
# the characters addresses, prices, file paths, code and emoticons are made of, so a font
# that stopped short of them dropped a glyph in the middle of ordinary writing.
ASCII_PUNCTUATION = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

# The marks everyday Korean writing needs that ASCII has no room for: the won sign, the
# reference mark that heads a note in almost every Korean form or notice, and the degree
# sign. Both candidate guide fonts carry all three.
KOREAN_DAILY_MARKS = list("\u20a9\u203b\u00b0")

# Deliberately absent, because refine_metrics derives each one from a mark the writer has
# already drawn — asking for them again would cost cells and buy nothing:
#   ‘ ’ from '     “ ” from "     – — ― − from -     … and · from .
SPECIAL_CHARS = ASCII_PUNCTUATION + KOREAN_DAILY_MARKS

# Everything produced by tracing the user's own handwriting (not the model), in order.
EXTRA_CHARS = ENG_CHARS + SPECIAL_CHARS

# The v4 sheet carried 125 guides on the same 130-cell page, so its last five cells were
# printed but empty. v5 fills all 130. That difference is the only way to tell a filled-in
# v4 sheet from a v5 one after cropping — both crop to 130 images — and telling them apart
# matters, because the first 33 cells of a v4 sheet are not the characters this version
# thinks they are. Used by font_processor._validated_cleaned_paths().
PREVIOUS_LAYOUT_GUIDE_COUNT = 125

KOR_STYLE_CELL_COUNT = len(KOR_STYLE_CHARS)
FULL_TEMPLATE_GUIDE_COUNT = KOR_STYLE_CELL_COUNT + len(EXTRA_CHARS)
FULL_TEMPLATE_CELL_COUNT = (
    math.ceil(FULL_TEMPLATE_GUIDE_COUNT / TEMPLATE_CELLS_PER_PAGE) *
    TEMPLATE_CELLS_PER_PAGE
)
ALLOWED_TEMPLATE_CELL_COUNTS = {KOR_STYLE_CELL_COUNT, FULL_TEMPLATE_CELL_COUNT}


def split_cells(cleaned_paths):
    """Split ordered cleaned-cell image paths into (style_imgs, traced_glyphs).

    Args:
        cleaned_paths: image paths sorted in template-cell order.
    Returns:
        style_imgs: list of paths for the Korean style references (<= 33).
        traced_glyphs: list of (char, path) for ENG/special glyphs to embed directly.
    """
    n_style = len(KOR_STYLE_CHARS)
    style_imgs = cleaned_paths[:n_style]
    extra_imgs = cleaned_paths[n_style:]
    traced_glyphs = list(zip(EXTRA_CHARS, extra_imgs))  # truncates to shortest
    return style_imgs, traced_glyphs
