"""Run a freshly built bundle's own smoke check and report the result.

Locating the executable inside the build output is the only part that differs by
platform, so that is all this does — the checks themselves live in desktop.py's
_smoke(), which runs inside the packaged app and therefore tests the thing that
actually ships rather than the checkout it was built from.

    python packaging/smoke_test.py "dist/Soul Font.app"
    python packaging/smoke_test.py dist/SoulFont
"""
import os
import subprocess
import sys
import tempfile


def find_executable(bundle):
    """The binary to run, given whatever PyInstaller produced on this platform."""
    if bundle.endswith(".app"):                       # macOS BUNDLE
        macos = os.path.join(bundle, "Contents", "MacOS")
        names = os.listdir(macos)
        return os.path.join(macos, names[0])
    if os.path.isdir(bundle):                         # COLLECT directory
        for name in ("SoulFont.exe", "SoulFont"):
            path = os.path.join(bundle, name)
            if os.path.isfile(path):
                return path
        sys.exit(f"[smoke] no SoulFont executable in {bundle}: {os.listdir(bundle)[:20]}")
    if os.path.isfile(bundle):
        return bundle
    sys.exit(f"[smoke] bundle not found: {bundle}")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    bundle = sys.argv[1]
    exe = find_executable(bundle)
    print(f"[smoke] running {exe}")

    env = dict(os.environ)
    env["SOULFONT_SMOKE"] = "1"
    # A scratch data dir, so the check never reads or writes a real install's database.
    env["SOULFONT_DATA_DIR"] = tempfile.mkdtemp(prefix="soulfont-smoke-")

    try:
        proc = subprocess.run([exe], env=env, timeout=180,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        sys.exit("[smoke] FAILED: the bundle did not finish within 180s")

    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(f"[smoke] FAILED: exit {proc.returncode}")
    print("[smoke] bundle OK")


if __name__ == "__main__":
    main()
