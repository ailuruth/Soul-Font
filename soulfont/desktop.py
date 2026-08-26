"""Soul Font as a desktop app.

The UI is already a web app, so the desktop build does not reimplement it: this starts
Django on a loopback port and puts the OS's own webview in front of it — WebKit on macOS,
WebView2 on Windows. The System 7 desktop the user sees is the same one the site serves,
which is why there is no second UI to keep in step.

Three things it has to get right that a dev server does not:

  writable state   BASE_DIR is inside the bundle and read-only, so the database, uploads
                   and build scratch go to the platform's per-user directory. See
                   apppaths.py.
  a real server    Django's runserver is single-threaded and explicitly not for this.
                   waitress is pure Python and works the same on both platforms, which
                   gunicorn (Unix only) does not.
  a free port      Hard-coding 8000 collides with whatever else the user is running, so
                   the port is taken from the OS and handed to the window.

Run from a checkout with:  python desktop.py
"""
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_ROOT)

# Must be set before Django settings are imported: they switch DEBUG, ALLOWED_HOSTS and
# the secret key on it.
os.environ["SOULFONT_APP"] = "1"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import apppaths  # noqa: E402

WINDOW_TITLE = "Soul Font"


def _free_port():
    """A port the OS says is free. Bound and released, so there is a small race, but the
    alternative is a fixed port that collides with the user's own dev server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _prepare():
    """Create the writable tree and bring the database up to date.

    Migrations run on every launch, not just the first: an update ships new migrations and
    the user's existing database in DATA_ROOT has to catch up before the UI opens.
    """
    apppaths.ensure_data_dirs()

    import django
    django.setup()
    from django.core.management import call_command

    call_command("migrate", interactive=False, verbosity=0)

    # Builds run on background threads, so any that was mid-flight when the app last quit
    # is gone. Reclaim those rows before the UI can show them, or they sit at a
    # percentage forever with no way to retry.
    from pybo.models import reclaim_interrupted_builds

    reclaimed = reclaim_interrupted_builds()
    if reclaimed:
        print(f"[desktop] reclaimed {reclaimed} interrupted build(s) from the last session")


def _serve(port):
    from waitress import serve
    from config.wsgi import application

    # threads: the font build runs on a background thread and holds a request open for a
    # long time; one thread would make the UI unresponsive for the whole build.
    serve(application, host="127.0.0.1", port=port, threads=8, _quiet=True)


def _wait_until_up(port, timeout=30.0):
    """Block until the server answers, so the window never opens on a connection error."""
    url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1).read(1)
            return True
        except urllib.error.HTTPError:
            return True          # answering at all is enough; 404 or 302 is still up
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.1)
    return False


def _smoke(port):
    """Prove a built bundle works without opening a window.

    A CI runner has no display, so the packaged app cannot be started normally there —
    but the failures worth catching (a data file PyInstaller did not collect, a Django
    app it could not import, a missing libpdfium) all happen before the window opens.
    This exercises exactly that much and exits.
    """
    import urllib.request

    checks = ["/", "/pybo/", "/login/", "/static/vendor/system/system.css"]
    failed = []
    for path in checks:
        try:
            status = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10).status
        except Exception as exc:                      # noqa: BLE001 - report, do not raise
            status = repr(exc)
        ok = status == 200
        print(f"[smoke] {'ok  ' if ok else 'FAIL'} {path} -> {status}")
        if not ok:
            failed.append(path)

    # The model has to be inside the bundle, not merely importable.
    from foundry.generate import resolve_backend
    backend, onnx_dir = resolve_backend()
    print(f"[smoke] backend={backend} dir={onnx_dir}")
    if backend != "onnx":
        failed.append("onnx model missing from bundle")
    else:
        # And it has to run. Finding the .onnx files proves PyInstaller collected them;
        # it says nothing about whether onnxruntime can load its own native libraries
        # from inside the bundle, which is the failure that would otherwise only show up
        # when a user waited through a build and got nothing.
        try:
            import numpy as np

            from foundry.char_layout import KOR_STYLE_CHARS
            from foundry.onnx_backend import OnnxGenerator, _decompose

            gen = OnnxGenerator(onnx_dir)
            style = np.zeros((len(KOR_STYLE_CHARS), 1, 128, 128), np.float32)
            comps = np.array([_decompose(c) for c in KOR_STYLE_CHARS], np.int64)
            ids = np.zeros(len(KOR_STYLE_CHARS), np.int64)
            gen.encode_write(ids, comps, style)
            out = gen.read_decode(np.zeros(1, np.int64),
                                  np.array([_decompose("한")], np.int64))
            if out.shape != (1, 1, 128, 128):
                raise ValueError(f"unexpected glyph shape {out.shape}")
            print(f"[smoke] ok   model ran, glyph {out.shape} "
                  f"range {out.min():.2f}..{out.max():.2f}")
        except Exception as exc:                      # noqa: BLE001 - report, do not raise
            print(f"[smoke] FAIL model did not run: {exc!r}")
            failed.append("onnx model failed to run")

    if failed:
        print(f"[smoke] FAILED: {failed}")
        return 1
    print("[smoke] all checks passed")
    return 0


def main():
    print(f"[desktop] data dir: {apppaths.DATA_ROOT}")
    _prepare()

    port = _free_port()
    threading.Thread(target=_serve, args=(port,), daemon=True).start()

    if not _wait_until_up(port):
        sys.exit(f"[desktop] server did not come up on port {port}")
    print(f"[desktop] serving on 127.0.0.1:{port}")

    if os.environ.get("SOULFONT_SMOKE") == "1":
        sys.exit(_smoke(port))

    import webview

    webview.create_window(WINDOW_TITLE, f"http://127.0.0.1:{port}/",
                          width=1180, height=820, min_size=(900, 640))
    # http_server=False: the app is the server. private_mode=False keeps the login
    # session across launches, which is what a desktop app is expected to do.
    webview.start(private_mode=False)


if __name__ == "__main__":
    main()
