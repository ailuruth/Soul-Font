# Soul Font

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.2-092E20?style=flat&logo=django&logoColor=white)](https://www.djangoproject.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

Soul Font turns a filled-in PDF template into an installable Korean font. You hand-write 33
Korean syllables — all of them ordinary words — and it generates the rest of Hangul, traces
the result into outlines, and gives you a Light/Regular/Bold TTF family.

![Soul Font — the desktop with Fonts, Letter and a font's page open](docs/DEMO.png)

## Download

[**Get the app**](https://github.com/nobel2040ne/Soul-Font/releases/latest) — macOS (Apple
Silicon) and Windows. Nothing else to install; the model ships inside it. Builds are
unsigned, so both systems warn on first launch — the release notes say how to get past it.

To run it from source instead, see [Setup](#setup).

Demo: https://www.youtube.com/watch?v=X25yWyhzacM

## Setup

Needs Python 3.9. No system packages — PDF rendering is bundled in the wheel.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd soulfont && python manage.py migrate
```

Then download the pretrained Korean generator from the
[DM-Font v1.0.0 release](https://github.com/clovaai/dmfont/releases/tag/v1.0.0) and save it
as `soulfont/checkpoints/korean-handwriting.pth`. It is too large for Git and generation
fails without it; everything else ships with the repo.

## Run

```bash
cd soulfont
python manage.py runserver      # http://127.0.0.1:8000/
```

The site opens on a System 7 desktop where each application is an icon you double-click.
Sign up, open **New Font**, download the blank template, fill it in by hand and upload the
PDF. The font's own page shows build progress — every font is the full 11,172 syllables, so
give it time — then downloads the family as a zip. From there you can rename it and set the
author, licence and copyright written into the file.

There is one interface at every width. A narrow screen gets the same desktop, with icons and
forms behaving as the plain links and forms they are in the markup.

## Desktop app

Soul Font also runs as a desktop app on macOS and Windows, with no Python or web server
for the user to set up.

```bash
pip install -r requirements.txt -r requirements-desktop.txt
cd soulfont
python scripts/export_onnx.py     # needs the checkpoint; writes onnx/ (~192 MB)
python desktop.py
```

It is the same site: Django on a loopback port, with the OS's own webview in front of it
(WebKit on macOS, WebView2 on Windows). There is no second UI.

Two things differ from the checkout:

- **The model runs on ONNX Runtime, not PyTorch.** `scripts/export_onnx.py` splits the
  generator into an encoder and a decoder graph, with the component memory bank in numpy
  — see `foundry/onnx_backend.py`. That takes ~310 MB out of the bundle and is faster on
  a machine with no GPU, which is most Windows machines.
  `scripts/verify_onnx.py` checks the graphs still reproduce the PyTorch model.
- **Writable state moves out of the project.** The bundle is read-only, so the database,
  uploads and build scratch go to `~/Library/Application Support/Soul Font` (macOS) or
  `%LOCALAPPDATA%\Soul Font` (Windows). `apppaths.py` decides; `SOULFONT_DATA_DIR`
  overrides it.

`SOULFONT_BACKEND=torch|onnx` forces a backend. Unset, ONNX is used when `onnx/` exists.

### Building installers

```bash
pyinstaller packaging/soulfont.spec --noconfirm
```

**PyInstaller does not cross-compile** — the `.app` must be built on macOS and the `.exe`
on Windows. `.github/workflows/desktop.yml` runs both on their own runners, exports the
model from the upstream checkpoint, verifies it against PyTorch, and smoke-tests each
bundle before uploading it. Builds are unsigned: macOS will show Gatekeeper's warning and
Windows SmartScreen's until they are signed and notarized.

## How it works

```text
crop the template -> generate Hangul (DM-Font) -> clean the rasters
                  -> trace to Béziers -> fit metrics -> write the TTF
```

`soulfont/foundry/` holds that pipeline, one file per step and in that order;
`soulfont/pybo/` is the Django app; `dmfont/` is the upstream model code.
[ARCHITECTURE.md](ARCHITECTURE.md) has the data model, the stages in detail and the disk
layout. Two knobs are
worth knowing: `SOUL_FONT_DEVICE=cpu` forces CPU inference and makes runs byte-reproducible,
and `SOULFONT_STROKE_EVENNESS` (default `0.6`, `0` to disable) controls how much the model's
uneven stroke width is smoothed out.

## Troubleshooting

- Generation fails immediately — the checkpoint is not at
  `soulfont/checkpoints/korean-handwriting.pth`.
- No font appears — it builds in a background thread, so the reason is in the server log.

Defaults are for local development: SQLite, `DEBUG = True`, and no password validation.

## Licence

Soul Font is open source under the [MIT License](LICENSE).

The third-party components (DM-Font, system.css, Galmuri, the Python dependencies) keep
their own terms, recorded in their per-file copyright headers.
