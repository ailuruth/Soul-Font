# Third-Party Notices

Soul Font itself is licensed under [PolyForm Noncommercial 1.0.0](LICENSE). The
components listed here are **not** covered by that licence — they are the property
of their own authors and carry their own terms, all of which survive Soul Font's
relicensing. Their notices must travel with any copy of Soul Font you distribute.

Last reviewed: 2026-08-26.

---

## DM-Font — the generation model

`soulfont/dmfont/`, `soulfont/models/`, `soulfont/utils/`, `soulfont/datasets/`,
`soulfont/meta/`, `soulfont/config/`

> DMFont
> Copyright (c) 2020-present NAVER Corp.
> MIT license
> https://github.com/clovaai/dmfont

MIT. Commercial use is permitted; the copyright and permission notice must be
retained, which the per-file headers already do.

### Exception: `soulfont/models/modules/modules.py`

Upstream DM-Font notes one exception to its MIT grant — `modules.py` is adopted
from [NVlabs/FUNIT](https://github.com/NVlabs/FUNIT), which is licensed
**CC BY-NC-SA 4.0 (NonCommercial)**. In this repository the FUNIT-derived portion
was the `weights_init` function, which has been **removed** (see the note in that
file). It was reachable only from `soulfont/dmfont/train.py`, which is not part of
the product path — Soul Font runs inference against a downloaded checkpoint and
never trains. The rest of `modules.py` is NAVER's own MIT-licensed code.

Do not reintroduce `weights_init` from upstream. Its licence forbids commercial use.

### The pretrained checkpoint — read this before selling

`soulfont/checkpoints/korean-handwriting.pth` is **not distributed with this
repository** and is not covered by any licence granted here. It is downloaded by
the operator from the
[DM-Font v1.0.0 release](https://github.com/clovaai/dmfont/releases/tag/v1.0.0).

Per DM-Font's own README, the Korean-handwriting model was trained on **UhBee
fonts** (어비 폰트). UhBee's published terms permit broad commercial *use* of the
fonts but state that selling the fonts themselves is prohibited, and that
conversion results (APK, WebFont and similar) must be distributed free of charge.

Whether glyphs the model generates in an end user's own handwriting style
constitute a "conversion result" of UhBee's fonts is an unsettled question, and
this repository takes no position on it. **This is a question for a lawyer, not
for this file.** Two ways to remove the doubt entirely:

- Obtain written clearance from the rights holder — <font@uhbee.net>.
- Train a replacement checkpoint on fonts you own or have licensed for the
  purpose, and ship that instead. `soulfont/scripts/prepare_dataset.py` and
  `soulfont/dmfont/train.py` are the path, though note that training needs CUDA
  and that the FUNIT initialiser has been removed from it.

---

## Bundled fonts

| File | Font | Licence | Notes |
|---|---|---|---|
| `soulfont/static/vendor/galmuri/*.ttf` | Galmuri 9/11/14 — © 2019-2025 Lee Minseo (quiple@quiple.dev) | SIL OFL 1.1 | Subset for this project; no outlines altered. See `NOTICE.txt` in that directory. OFL permits commercial embedding; it forbids selling the font on its own. |
| `soulfont/media/ttf_files/MaruBuri-Regular.ttf` | MaruBuri — © NAVER Corp., © NAVER Cultural Foundation | SIL OFL 1.1 | Same conditions as above. |
| `soulfont/static/vendor/system/monaco.woff{,2}` | Monaco | **Apple proprietary — unresolved** | Monaco is Apple's typeface. It is not known to be licensed for redistribution by third parties. Serving it from a commercial site is the sharpest licensing exposure in this repository. Replace it or clear it. |
| `soulfont/static/vendor/system/ChicagoFLF.woff{,2}` | ChicagoFLF | Freeware — **terms unverified** | A third-party recreation of Apple's Chicago, redistributed via system.css. Confirm the author's terms before commercial deployment. |
| `soulfont/static/vendor/system/ChiKareGo2.woff{,2}` | ChiKareGo2 | Freeware — **terms unverified** | Chicago 12pt recreation. Same caveat. |
| `soulfont/static/vendor/system/FindersKeepers.woff{,2}` | FindersKeepers | Freeware — **terms unverified** | Geneva 9pt recreation. Same caveat. |

The four `system/` faces ship as part of system.css (below). "Freeware" as
commonly distributed is not the same as "licensed for redistribution inside a
paid product" — each needs its author's actual terms read before launch.

---

## Web and JavaScript components

**system.css** — `soulfont/static/vendor/system/system.css` and its SVGs
Copyright (c) 2022 Sakun Acharige. MIT. https://github.com/sakofchit/system.css
(The fonts it references are listed separately above; MIT covers the CSS, not them.)

**html2canvas** — `soulfont/static/vendor/html2canvas/html2canvas.min.js`
Copyright (c) Niklas von Hertzen. MIT. https://github.com/niklasvh/html2canvas

**ImageTracer.js** — `soulfont/public/javascript/imagetracer_v1.2.1.js`
By András Jankovics <andras@jankovics.net>. Released to the public domain under
the Unlicense. https://github.com/jankovicsandras/imagetracerjs

**jQuery 3.4.1** — `soulfont/public/javascript/jquery-3.4.1.js`
Copyright OpenJS Foundation and other contributors. MIT. https://jquery.com

**svg2ttf (vendored file)** — `soulfont/public/javascript/os2.js`
A copy of `lib/sfnt/os2.js` from svg2ttf. Copyright (c) 2013-2014 Vitaly Puzrin.
MIT. https://github.com/fontello/svg2ttf

### Desktop build

Installed from `requirements-desktop.txt`; these ship inside the packaged app.

| Package | Licence | Role |
|---|---|---|
| pywebview | BSD-3-Clause | window shell around the OS webview |
| waitress | ZPL-2.1 | WSGI server |
| whitenoise | MIT | serves static files from inside the app |
| onnxruntime | MIT | runs the exported model in place of PyTorch |
| pyinstaller | GPL-2.0 **with a bundling exception** | build tool |

PyInstaller's licence is the one to understand. It is GPL-2.0 with an explicit exception
permitting the bundling of applications under any licence, so using it to package Soul
Font does not make Soul Font GPL. What must not happen is redistributing a modified
PyInstaller itself. The bootloader shipped inside the bundle is covered by that same
exception.

The exported model in `onnx/` carries whatever terms the DM-Font checkpoint it was
converted from carries — converting a format does not change the licence of the weights.
See **The pretrained checkpoint** above; the UhBee question applies to the desktop build
exactly as it does to the web app.

### Node dependencies of `soulfont/foundry/generateTTF.js`

Installed from `soulfont/package.json`; not vendored in git, but they ship with
any deployment. All permissive, all requiring notice retention:

| Package | Licence |
|---|---|
| canvas | MIT |
| express | MIT |
| fonteditor-core | MIT (© 2014 ecomfe) |
| graceful-fs | ISC |
| imagetracerjs | Unlicense |
| opentype.js | MIT |
| pngjs | MIT |
| svg2ttf | MIT |
| svgicons2svgfont | MIT |

---

## Python dependencies

Installed from `requirements.txt`; not vendored in git. Django (BSD-3-Clause),
PyTorch (BSD-3-Clause), NumPy (BSD-3-Clause), Pillow (MIT-CMU), fontTools (MIT),
scikit-image (BSD-3-Clause), SciPy (BSD-3-Clause), matplotlib (PSF-based),
h5py (BSD-3-Clause), PyYAML (MIT), sconf (MIT), fire (Apache-2.0),
tqdm (MPL-2.0 / MIT), tensorboardX (MIT). All permit commercial use.

**pypdfium2 / PDFium** — `pypdfium2==5.13.0`, BSD-3-Clause and Apache-2.0.
Renders the uploaded PDF. The wheel bundles `libpdfium` itself (`.dylib` on macOS,
`.dll` on Windows), so there is no system package to install and nothing copyleft
to redistribute — which is what makes a packaged desktop build possible.

This replaced pdf2image/Poppler. Poppler is GPL-2.0/GPL-3.0; invoking it as a
separate executable was arm's-length and fine, but shipping it inside a paid
installer would have meant distributing GPL binaries under GPL terms. If you ever
reintroduce it, that distinction is the one that matters.
