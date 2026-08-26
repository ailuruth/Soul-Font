"""Where the app reads from and where it writes to.

Running from a checkout these are the same directory and nothing changes. Running from a
packaged desktop build they cannot be: the code, the templates and the exported model sit
inside the .app bundle or under Program Files, which is read-only and — on macOS —
signed, so writing there breaks the signature. The database, the uploaded PDFs, the
build scratch space and the finished fonts have to live in the user's own data
directory instead.

    CODE_ROOT   read-only    code, templates, static/, data/charset/, onnx/, checkpoints/
    DATA_ROOT   writable     db.sqlite3, media/, uploads/, workdir/

SOULFONT_DATA_DIR sets DATA_ROOT. The desktop launcher points it at the platform's
per-user location; unset, it falls back to CODE_ROOT, which is the checkout behaviour and
keeps every existing path exactly where it was.
"""
import os
import sys

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))


def _frozen():
    """True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def default_data_dir():
    """The per-user writable location for this platform."""
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/Soul Font")
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(base, "Soul Font")
    # Linux and anything else: the XDG spec's data home.
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "soulfont")


def _resolve_data_root():
    explicit = os.environ.get("SOULFONT_DATA_DIR", "").strip()
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    # A frozen build must never write into the bundle, so it gets the per-user directory
    # whether or not the launcher remembered to set the variable.
    return default_data_dir() if _frozen() else CODE_ROOT


DATA_ROOT = _resolve_data_root()


def data_path(*parts):
    return os.path.join(DATA_ROOT, *parts)


def code_path(*parts):
    return os.path.join(CODE_ROOT, *parts)


def ensure_data_dirs():
    """Create the writable tree. Safe to call repeatedly."""
    for rel in ("media", "media/ttf_files", "uploads", "workdir",
                "workdir/crops", "workdir/glyphs", "workdir/configs", "workdir/fonts"):
        os.makedirs(data_path(*rel.split("/")), exist_ok=True)
    return DATA_ROOT


def install_secret_key():
    """This install's Django SECRET_KEY, created on first run and kept in DATA_ROOT.

    A signing key baked into a shipped binary is the same key on every machine that
    installs it, which would let anyone holding the app forge session cookies for anyone
    else's. Generating one per install costs nothing here — the app is single-user and
    local — and removes that.
    """
    import secrets
    import stat

    path = data_path("secret_key")
    try:
        with open(path, encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            return key
    except FileNotFoundError:
        pass

    key = secrets.token_urlsafe(64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(key)
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)   # owner-only; no-op on Windows
    except OSError:
        pass
    return key
