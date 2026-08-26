"""Glyph generation entry point: picks a backend and runs it.

Two backends produce the same glyphs from the same weights:

    onnx   ONNX Runtime, via foundry.onnx_backend. No torch. This is what the packaged
           desktop build ships, because torch is 312 MB against onnxruntime's ~60 MB and
           is markedly slower on a CPU-only machine.
    torch  foundry.inference, the original. Still the path for training, for CUDA/MPS,
           and for anyone running from a checkout who has not exported the graphs.

The choice is made here rather than inside foundry.inference because that module imports
torch at the top: a build whose point is not shipping torch cannot import it to ask which
backend to use. So torch is imported only on the branch that needs it.

SOULFONT_BACKEND=torch|onnx forces one. Unset, ONNX is used when onnx/ has been exported
(see scripts/export_onnx.py) and torch otherwise.
"""
import os
import re
import threading

import yaml

from foundry.onnx_backend import onnx_dir_if_present

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# One generation at a time. The web app builds fonts on background threads, and both
# backends hold a lot of memory while running; two at once is how a machine runs out.
_LOCK = threading.Lock()


def _hangul_targets(target_chars):
    """The model only knows Hangul; everything else is traced from the writer's own hand."""
    return list(dict.fromkeys(c for c in target_chars if re.match(r'[가-힣]', c)))


def resolve_backend():
    """-> ('onnx', dir) or ('torch', None)."""
    requested = os.environ.get("SOULFONT_BACKEND", "").strip().lower()
    onnx_dir = onnx_dir_if_present(PROJECT_ROOT)

    if requested == "torch":
        return "torch", None
    if requested == "onnx":
        if not onnx_dir:
            raise RuntimeError(
                "SOULFONT_BACKEND=onnx but onnx/ has no exported model. "
                "Run: python scripts/export_onnx.py"
            )
        return "onnx", onnx_dir
    if requested:
        raise ValueError(f"SOULFONT_BACKEND must be 'torch' or 'onnx', got {requested!r}")

    return ("onnx", onnx_dir) if onnx_dir else ("torch", None)


def _save_png(glyph, path):
    """glyph: [1, 128, 128] float32 in [-1, 1] -> grayscale PNG."""
    import numpy as np
    from PIL import Image

    arr = (glyph[0] * 0.5 + 0.5).clip(0, 1) * 255.0
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _run_onnx(cfg, onnx_dir, save_dir, device_name=None, progress=None, batch_size=128):
    from foundry.onnx_backend import OnnxGenerator, decode_batches, style_batch

    # "CPU only" has to mean it, or the setting is decoration.
    providers = ["CPUExecutionProvider"] if device_name == "cpu" else None
    gen = OnnxGenerator(onnx_dir, providers=providers)
    print(f"[generate] backend=onnx providers={gen.providers}")

    style_ids, comp_ids, imgs = style_batch(cfg['style_imgs'], cfg['style_chars'])
    gen.encode_write(style_ids, comp_ids, imgs)

    targets = _hangul_targets(cfg['target_chars'])
    total = len(targets)
    written = 0
    for trg_ids, trg_comps in decode_batches(targets, batch_size=batch_size):
        outs = gen.read_decode(trg_ids, trg_comps)
        for glyph in outs:
            path = os.path.join(save_dir, f"inferred_{ord(targets[written]):04X}.png")
            _save_png(glyph, path)
            written += 1
        if progress:
            progress(written, total)
    return written


def run(config, checkpoint, save_dir, device_name=None, use_amp=None, progress=None):
    """Generate every Hangul target in `config` into `save_dir` as inferred_XXXX.png.

    progress: optional callable(done, total), called as batches finish. This phase is the
    longest part of a build — minutes of it — so without a callback the UI has nothing to
    show and looks stalled.
    """
    os.makedirs(save_dir, exist_ok=True)
    backend, onnx_dir = resolve_backend()

    if backend == "torch":
        from foundry.inference import main as torch_main   # imports torch
        print("[generate] backend=torch")
        with _LOCK:
            return torch_main(config, checkpoint, save_dir,
                              device_name=device_name, use_amp=use_amp)

    with open(config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    with _LOCK:
        n = _run_onnx(cfg, onnx_dir, save_dir, device_name=device_name, progress=progress)
    print(f"[generate] saved {n} glyphs to {save_dir}")
    return n


def describe_acceleration():
    """What acceleration is actually usable here, for the UI to describe honestly.

    Returns {available, label, detail}. The UI used to promise "uses the GPU when there
    is one, about 4x faster" to everyone, including the many Windows machines with no
    usable GPU and the packaged build, which runs ONNX Runtime on CPU. Offering a choice
    that does nothing, and quoting a speedup that will not happen, is worse than offering
    nothing: the user waits for a build they think is running fast.
    """
    backend, onnx_dir = resolve_backend()

    if backend == "onnx":
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
        except Exception:                             # noqa: BLE001
            providers = []
        if "CUDAExecutionProvider" in providers:
            return {"available": True, "backend": backend, "label": "NVIDIA GPU (CUDA)",
                    "detail": "Noticeably faster than the CPU, with the same glyphs."}
        # CoreML is present on every Mac but is deliberately not used - it falls back
        # per-op and has changed this graph's output, and a font that differs by machine
        # is not acceptable. So on a Mac the honest answer here is still CPU.
        return {"available": False, "backend": backend, "label": "CPU",
                "detail": "This machine has no GPU Soul Font can use, so it runs on the "
                          "CPU. Everything works; it just takes longer."}

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return {"available": True, "backend": backend, "label": f"NVIDIA GPU ({name})",
                    "detail": "About 4x faster than the CPU, with the same glyphs."}
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return {"available": True, "backend": backend, "label": "Apple Silicon (MPS)",
                    "detail": "About 4x faster than the CPU, with the same glyphs."}
    except Exception:                                 # noqa: BLE001
        pass
    return {"available": False, "backend": backend, "label": "CPU",
            "detail": "This machine has no GPU Soul Font can use, so it runs on the CPU. "
                      "Everything works; it just takes longer."}
