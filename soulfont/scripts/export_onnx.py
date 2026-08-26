"""Export the DM-Font generator to ONNX, as two graphs plus a memory bank.

Why two graphs. MACore is not one forward pass — it is stateful. `encode_write` runs the
component encoder over the style images and writes each component's features into a
memory bank; `read_decode` reads that bank back and decodes a target syllable from it.
The bank is a plain Python dict keyed by (style, component) holding a list of tensors that
are averaged on read (models/memory.py). Dicts do not trace, and they do not need to: the
bank holds no learned weights, so it belongs in the runtime, not in the graph.

So the split follows the model's own seam:

    encoder.onnx    style images  ->  component features + skip features
    (numpy)         average features per component into the bank
    decoder.onnx    bank read + component ids  ->  the glyph

The persistent memory — a learned per-component bias plus a hypernet, applied on read —
is a real module with weights, so it goes inside decoder.onnx rather than the runtime.
Only the dynamic bank is numpy.

Usage:
    python scripts/export_onnx.py [--checkpoint PATH] [--out-dir onnx] [--opset 17]
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from foundry.inference import load_model, get_device  # noqa: E402

MODEL_CFG = dict(C=32, n_comps=68, n_comp_types=3, language="kor")


class Encoder(nn.Module):
    """Style image -> (component features, skip features).

    The component encoder returns a list: feats[0] is the shared trunk output and
    feats[1:] are the per-head stacks. MACore takes feats[-1] as the component features
    and filter_skips() picks the skip level out of the same list. Both are needed, so
    both come out here rather than re-running the trunk twice.
    """

    def __init__(self, component_encoder):
        super().__init__()
        self.component_encoder = component_encoder

    def forward(self, style_imgs):
        feats = self.component_encoder(style_imgs)
        comp_feats = feats[-1]
        skips = self.component_encoder.filter_skips(feats)
        return comp_feats, skips[0]


class Decoder(nn.Module):
    """(bank read, component addresses, skip read) -> glyph.

    Mirrors Memory.read + MACore.read_decode. The main bank read goes through the
    persistent memory first; the skip bank is non-persistent (skip_memory is built with
    persistent=False) so its read is passed straight through, exactly as MACore does.
    """

    def __init__(self, persistent_memory, decoder):
        super().__init__()
        self.persistent_memory = persistent_memory
        self.decoder = decoder

    def forward(self, mem, comp_addrs, skip_feats):
        mem = self.persistent_memory(mem, comp_addrs)
        return self.decoder(mem, [skip_feats])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint",
                    default=os.path.join(PROJ_ROOT, "checkpoints", "korean-handwriting.pth"))
    ap.add_argument("--out-dir", default=os.path.join(PROJ_ROOT, "onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"[export] checkpoint not found: {args.checkpoint}")

    device = get_device("cpu")          # export on CPU so the graph carries no device ops
    gen = load_model(args.checkpoint, MODEL_CFG, device)
    gen.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- shapes, measured rather than assumed -------------------------------------
    probe = torch.zeros(2, 1, 128, 128)
    with torch.no_grad():
        comp_feats, skip_feats = Encoder(gen.component_encoder)(probe)
    print(f"[export] comp_feats {tuple(comp_feats.shape)}  skip_feats {tuple(skip_feats.shape)}")

    # --- encoder ------------------------------------------------------------------
    enc_path = os.path.join(args.out_dir, "encoder.onnx")
    torch.onnx.export(
        Encoder(gen.component_encoder), (probe,), enc_path,
        input_names=["style_imgs"], output_names=["comp_feats", "skip_feats"],
        dynamic_axes={"style_imgs": {0: "B"},
                      "comp_feats": {0: "B"}, "skip_feats": {0: "B"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"[export] wrote {enc_path} ({os.path.getsize(enc_path)/1e6:.1f} MB)")

    # --- decoder ------------------------------------------------------------------
    dec_path = os.path.join(args.out_dir, "decoder.onnx")
    mem = torch.zeros_like(comp_feats)
    skip = torch.zeros_like(skip_feats)
    addrs = torch.zeros(2, MODEL_CFG["n_comp_types"], dtype=torch.long)
    torch.onnx.export(
        Decoder(gen.memory.persistent_memory, gen.decoder),
        (mem, addrs, skip), dec_path,
        input_names=["mem", "comp_addrs", "skip_feats"], output_names=["glyph"],
        dynamic_axes={"mem": {0: "B"}, "comp_addrs": {0: "B"},
                      "skip_feats": {0: "B"}, "glyph": {0: "B"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"[export] wrote {dec_path} ({os.path.getsize(dec_path)/1e6:.1f} MB)")

    # --- what the runtime needs to rebuild the bank --------------------------------
    meta = {
        "opset": args.opset,
        "language": MODEL_CFG["language"],
        "n_comps": MODEL_CFG["n_comps"],
        "n_comp_types": MODEL_CFG["n_comp_types"],
        "comp_feats_shape": list(comp_feats.shape[1:]),
        "skip_feats_shape": list(skip_feats.shape[1:]),
        "source_checkpoint": os.path.basename(args.checkpoint),
    }
    meta_path = os.path.join(args.out_dir, "model.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] wrote {meta_path}")
    total = sum(os.path.getsize(os.path.join(args.out_dir, f))
                for f in os.listdir(args.out_dir))
    print(f"[export] total {total/1e6:.1f} MB "
          f"(checkpoint was {os.path.getsize(args.checkpoint)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
