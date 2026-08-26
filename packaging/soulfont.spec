# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Soul Font desktop app. Builds on macOS and on Windows.

PyInstaller does not cross-compile: the .app has to be built on a Mac and the .exe on a
Windows machine. Both come from this one spec — see .github/workflows/desktop.yml, which
runs it on both runners.

    pyinstaller packaging/soulfont.spec --noconfirm

What needs saying explicitly, because PyInstaller's analysis cannot see it:

  Django            loads apps, middleware and migrations by name at runtime, so nothing
                    under pybo/ or config/ is reachable by following imports.
  templates/static  data, not modules.
  onnx/             the exported model. Bundled rather than downloaded so the app works
                    offline on first run.
  torch             deliberately excluded. The whole point of the ONNX backend is that
                    the desktop build does not carry 312 MB of it; if it is installed in
                    the build environment PyInstaller would otherwise pull it in.
"""
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
ROOT = os.path.dirname(SPEC_DIR)
APP = os.path.join(ROOT, "soulfont")

IS_MAC = sys.platform == "darwin"

# --- data ------------------------------------------------------------------------
datas = [
    (os.path.join(APP, "pybo", "templates"), "pybo/templates"),
    (os.path.join(APP, "static"), "static"),
    (os.path.join(APP, "data"), "data"),
    (os.path.join(APP, "meta"), "meta"),
    (os.path.join(APP, "config"), "config"),
    (os.path.join(APP, "onnx"), "onnx"),
]
# MaruBuri is the fallback face the metadata step measures against; it is the only file
# under media/ that ships rather than being produced.
_maruburi = os.path.join(APP, "media", "ttf_files", "MaruBuri-Regular.ttf")
if os.path.isfile(_maruburi):
    datas.append((_maruburi, "media/ttf_files"))

# The attribution notice has to reach whoever gets only the binary: the per-file
# copyright headers are in the source, and a packaged app has no source.
datas.append((os.path.join(ROOT, "NOTICE"), "."))

datas += collect_data_files("pypdfium2_raw")     # libpdfium
datas += collect_data_files("onnxruntime")

# --- imports Django resolves by name --------------------------------------------
hiddenimports = [
    "pybo", "pybo.urls", "pybo.views", "pybo.models", "pybo.apps", "pybo.admin",
    "config", "config.settings", "config.urls", "config.wsgi",
    "apppaths",
    "foundry.generate", "foundry.onnx_backend", "foundry.font_processor",
    "foundry.glyph_vectorizer", "foundry.set_font_metadata", "foundry.char_layout",
    "whitenoise", "whitenoise.middleware",
    "waitress",
]
hiddenimports += collect_submodules("django")
hiddenimports += collect_submodules("pybo.migrations")

a = Analysis(
    [os.path.join(APP, "desktop.py")],
    pathex=[APP],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # torch/torchvision: replaced by the ONNX backend, and the reason the build is ~310 MB
    # smaller. matplotlib and h5py are only used by template generation and training,
    # neither of which the app does. tkinter is pulled in by matplotlib's backends.
    excludes=["torch", "torchvision", "matplotlib", "h5py", "tensorboardX",
              "scipy.io.matlab", "tkinter", "PyQt5", "PySide2", "IPython", "pytest",
              # Not project dependencies. These arrive when the build environment has
              # extra packages installed alongside (opencv pulls cv2 at 111 MB, PyAV
              # pulls av at 50 MB) and PyInstaller follows them in. A clean CI venv would
              # not have them; naming them here means a developer's working venv produces
              # the same bundle CI does.
              "cv2", "av", "pandas", "boto3", "botocore", "lxml", "sympy",
              "onnx",            # the exporter, not the runtime
              ],
    noarchive=False,
)

# --- keep local state out of the bundle ------------------------------------------
#
# PyInstaller's Django hook sweeps the project directory for data, which picks up
# db.sqlite3 — the developer's database, with real accounts and font records in it. That
# must not ship: a buyer would receive it, and the app would then run against someone
# else's data instead of creating their own in DATA_ROOT. The same applies to anything
# under uploads/ or workdir/, which are other people's handwriting.
_LOCAL_STATE = ("db.sqlite3", "secret_key")
_LOCAL_DIRS = ("uploads/", "workdir/", "staticfiles/", "checkpoints/")


def _is_local_state(dest):
    norm = dest.replace(os.sep, "/")
    if os.path.basename(norm) in _LOCAL_STATE or norm.endswith(".sqlite3"):
        return True
    return any(norm == d.rstrip("/") or norm.startswith(d) for d in _LOCAL_DIRS)


_before = len(a.datas)
a.datas = TOC([entry for entry in a.datas if not _is_local_state(entry[0])])
_removed = _before - len(a.datas)
print(f"[spec] dropped {_removed} local-state file(s) from the bundle")

pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="SoulFont",
    debug=False,
    strip=False,
    upx=False,          # UPX breaks code signing on macOS and trips antivirus on Windows
    console=False,      # no terminal window behind the app
)

coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="SoulFont",
)

if IS_MAC:
    app = BUNDLE(
        coll,
        name="Soul Font.app",
        icon=None,
        bundle_identifier="dev.soulfont.app",
        info_plist={
            "CFBundleName": "Soul Font",
            "CFBundleDisplayName": "Soul Font",
            "CFBundleShortVersionString": "2.0.0",
            "CFBundleVersion": "2.0.0",
            "NSHighResolutionCapable": True,
            # The app talks only to its own loopback server; without this App Transport
            # Security blocks http://127.0.0.1 and the window comes up blank.
            "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
        },
    )
