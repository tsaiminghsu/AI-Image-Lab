"""Full vs quantized (fp8) variants of the SDXL/Pony checkpoints - converter and settings CLI.

Run with ComfyUI's venv python (torch/safetensors live only there):

    ComfyUI\\.venv\\Scripts\\python.exe training\\quantize_models.py convert --model cyberrealistic_pony
    ComfyUI\\.venv\\Scripts\\python.exe training\\quantize_models.py verify --model cyberrealistic_pony
    ComfyUI\\.venv\\Scripts\\python.exe training\\quantize_models.py status
    ComfyUI\\.venv\\Scripts\\python.exe training\\quantize_models.py set cyberrealistic_pony=quant

Output format: ComfyUI's per-layer quantized format, not a plain cast. A plain cast-to-fp8 checkpoint
is upcast back to fp16 by CheckpointLoaderSimple on this RTX 2070 (no fp8 compute), so it would save
disk but not VRAM. With per-layer markers ComfyUI keeps the quantized Linear layers in fp8
("emulated ops": stored fp8, dequantized to fp16 per layer in the forward pass). For each selected
Linear weight L:
    L.weight        float8_e4m3fn, clamp(w / scale, -448, 448)
    L.weight_scale  float32 0-dim, amax(|w|) / 448   (dequantize: q * scale)
    L.comfy_quant   uint8, UTF-8 bytes of {"format": "float8_e4m3fn"}
Only the transformer-block Linear layers (attention q/k/v/out, feed-forward) and proj_in/proj_out are
quantized - about 86% of the SDXL UNet parameters. Convs, norms, biases, time/label embeddings, the
text encoders and the VAE are copied bit-exact. The source file is only ever read (several
checkpoints are NTFS hardlinks shared with other folders); the result is a new <stem>.fp8q.safetensors
next to it.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import comfyui_client as client  # noqa: E402  (no torch import there)

UNET_PREFIX = "model.diffusion_model."
FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
QCONF_JSON = json.dumps({"format": "float8_e4m3fn"})
ALLOW_LIST_VERSION = 1
ALLOW_RE = re.compile(
    r"^(input_blocks|middle_block|output_blocks)\.\d+(\.\d+)*\.("
    r"transformer_blocks\.\d+\.(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)"
    r"|transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)"
    r"|proj_in|proj_out)\.weight$"
)
QUANT_SIZE_RATIO = 0.72  # measured estimate: quant file / full file for SDXL


def _paths(key):
    v = client.MODEL_VARIANTS[key]
    return os.path.join(client.CHECKPOINTS_DIR, v["full"]), os.path.join(client.CHECKPOINTS_DIR, v["quant"])


def _is_quant_target(key: str, tensor) -> bool:
    return key.startswith(UNET_PREFIX) and tensor.ndim == 2 and bool(ALLOW_RE.match(key[len(UNET_PREFIX):]))


def _quantize(w, device):
    import torch
    w32 = w.to(device, torch.float32)
    amax = w32.abs().max()
    scale = amax / FP8_MAX if float(amax) > 0 else torch.tensor(1.0, device=device)
    q = (w32 / scale).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.cpu().contiguous(), scale.float().reshape(()).cpu()


def convert(key: str, force: bool = False, device: str = "cpu", verify_after: bool = True) -> str:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = _paths(key)
    if not os.path.isfile(src):
        raise SystemExit(f"{key}: source checkpoint not found: {src}")
    if os.path.normcase(os.path.abspath(src)) == os.path.normcase(os.path.abspath(dst)):
        raise SystemExit(f"{key}: output path equals the source - refusing")
    if os.path.exists(dst):
        if not force:
            raise SystemExit(f"{key}: {dst} already exists (use --force to rebuild it)")
        if os.stat(dst).st_nlink > 1:
            raise SystemExit(f"{key}: {dst} is a hardlink shared with another path - refusing to overwrite")
    need = int(os.path.getsize(src) * QUANT_SIZE_RATIO) + (1 << 30)
    free = shutil.disk_usage(client.CHECKPOINTS_DIR).free
    if free < need:
        raise SystemExit(f"{key}: not enough disk space ({free / 1e9:.1f} GB free, need ~{need / 1e9:.1f} GB)")

    qconf = torch.tensor(list(QCONF_JSON.encode("utf-8")), dtype=torch.uint8)
    out, n_q, p_q, p_unet, b_in = {}, 0, 0, 0, 0
    with safe_open(src, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        keys = list(f.keys())
        if any(k.endswith(".comfy_quant") for k in keys):
            raise SystemExit(f"{key}: {src} is already quantized")
        for k in keys:
            t = f.get_tensor(k)
            b_in += t.numel() * t.element_size()
            if k.startswith(UNET_PREFIX):
                p_unet += t.numel()
            if _is_quant_target(k, t):
                base = k[: -len(".weight")]
                q, s = _quantize(t, device)
                out[k] = q
                out[base + ".weight_scale"] = s
                out[base + ".comfy_quant"] = qconf.clone()
                n_q += 1
                p_q += t.numel()
            else:
                out[k] = t.contiguous()
    if n_q == 0:
        raise SystemExit(f"{key}: no layer matched the allow-list - not an SDXL UNet?")

    meta["aiimagelab_quant"] = json.dumps({
        "source": os.path.basename(src), "source_bytes": os.path.getsize(src),
        "format": "float8_e4m3fn", "allow_list_version": ALLOW_LIST_VERSION,
        "date": datetime.date.today().isoformat(),
    })
    tmp = dst + ".partial"
    save_file(out, tmp, metadata={str(k): str(v) for k, v in meta.items()})
    os.replace(tmp, dst)
    b_out = os.path.getsize(dst)
    print(f"{key}: quantized {n_q} Linear layers, {p_q / 1e6:.0f}M of {p_unet / 1e6:.0f}M UNet params "
          f"({100 * p_q / max(p_unet, 1):.0f}%); {b_in / 1e9:.2f} GB -> {b_out / 1e9:.2f} GB", flush=True)
    print(f"wrote {dst}", flush=True)
    if verify_after:
        verify(key)
    return dst


def verify(key: str, samples: int = 20) -> None:
    """Check the file the way ComfyUI will read it, then compare dequantized weights to the source."""
    import torch
    from safetensors import safe_open

    src, dst = _paths(key)
    if not os.path.isfile(dst):
        raise SystemExit(f"{key}: {dst} not found - run convert first")
    comfy_dir = os.path.abspath(client.COMFYUI_DIR)
    if comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    import comfy.model_detection
    import comfy.utils

    sd = comfy.utils.load_torch_file(dst)
    problems = []
    prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
    if prefix != UNET_PREFIX:
        problems.append(f"unet prefix {prefix!r} != {UNET_PREFIX!r}")
    quant = comfy.utils.detect_layer_quantization(sd, prefix)
    if not quant or not quant.get("mixed_ops"):
        problems.append(f"ComfyUI does not detect the per-layer quantization (got {quant!r})")
    cfg = comfy.model_detection.model_config_from_unet(sd, prefix)
    if type(cfg).__name__ != "SDXL":
        problems.append(f"model config detected as {type(cfg).__name__}, expected SDXL")

    markers = [k for k in sd if k.endswith(".comfy_quant")]
    bad_markers = [k for k in markers if bytes(sd[k].tolist()).decode("utf-8", "replace") != QCONF_JSON]
    bad_scales = [k for k in sd if k.endswith(".weight_scale") and (sd[k].dtype != torch.float32 or sd[k].ndim != 0)]
    bad_weights = [k[: -len(".comfy_quant")] + ".weight" for k in markers
                   if sd[k[: -len(".comfy_quant")] + ".weight"].dtype != torch.float8_e4m3fn]
    for label, lst in (("marker payload", bad_markers), ("scale dtype/shape", bad_scales), ("weight dtype", bad_weights)):
        if lst:
            problems.append(f"{len(lst)} bad {label}, e.g. {lst[0]}")

    cos_min, rel_max = 1.0, 0.0
    if os.path.isfile(src) and markers:
        step = max(1, len(markers) // samples)
        with safe_open(src, framework="pt", device="cpu") as f:
            for m in markers[::step][:samples]:
                base = m[: -len(".comfy_quant")]
                w = f.get_tensor(base + ".weight").float().flatten()
                deq = (sd[base + ".weight"].float() * sd[base + ".weight_scale"]).flatten()
                cos_min = min(cos_min, float(torch.nn.functional.cosine_similarity(w, deq, dim=0)))
                rel_max = max(rel_max, float((w - deq).norm() / w.norm().clamp_min(1e-12)))
        if cos_min < 0.999:
            problems.append(f"dequantized weights too far from the source (min cosine {cos_min:.5f})")

    print(f"{key}: {len(markers)} quantized layers, prefix ok={prefix == UNET_PREFIX}, detected={quant}, "
          f"config={type(cfg).__name__}, sample min cosine {cos_min:.5f}, max relative RMS error {rel_max:.3%}",
          flush=True)
    if problems:
        raise SystemExit(f"{key}: VERIFY FAILED - " + "; ".join(problems))
    print(f"{key}: verify OK", flush=True)


def status() -> None:
    rows = client.variant_status()
    for r in rows:
        r["full_label"] = f"{r['full']}{'' if r['full_present'] else ' (missing)'}"
        r["quant_label"] = f"{r['quant']}{'' if r['quant_present'] else ' (not converted)'}"
    wf = max([len("full file")] + [len(r["full_label"]) for r in rows])
    wq = max([len("quant file")] + [len(r["quant_label"]) for r in rows])
    print(f"{'model':20s} {'chosen':6s}  {'full file':{wf}s} {'GB':>6s}  {'quant file':{wq}s} {'GB':>6s}")
    for r in rows:
        note = f"  unsupported: {r['unsupported']}" if r["unsupported"] else ""
        print(f"{r['key']:20s} {r['chosen']:6s}  {r['full_label']:{wf}s} {r['full_gb'] or 0:6.2f}  "
              f"{r['quant_label']:{wq}s} {r['quant_gb'] or 0:6.2f}{note}")
    print(f"settings file: {client.MODEL_VARIANTS_FILE}")


def set_choices(pairs, default=None) -> None:
    models = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"expected KEY=full|quant, got {pair!r}")
        k, v = pair.split("=", 1)
        if k not in client.MODEL_VARIANTS:
            raise SystemExit(f"unknown model {k!r} - choices: {', '.join(client.QUANT_MODEL_KEYS)}")
        if v not in client.VARIANT_CHOICES:
            raise SystemExit(f"unknown variant {v!r} - choices: full, quant")
        models[k] = v
    path = client.save_variant_settings(models=models, default=default)
    print(f"saved {path}")
    status()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="write <stem>.fp8q.safetensors for one or all SDXL/Pony checkpoints")
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", choices=client.QUANT_MODEL_KEYS)
    g.add_argument("--all", action="store_true")
    c.add_argument("--force", action="store_true", help="rebuild an existing quantized file")
    c.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="where to compute the scales (cpu is fine)")
    c.add_argument("--no-verify", action="store_true")
    v = sub.add_parser("verify", help="check a quantized file the way ComfyUI reads it")
    v.add_argument("--model", required=True, choices=client.QUANT_MODEL_KEYS)
    sub.add_parser("status", help="show both files and the chosen variant per model")
    s = sub.add_parser("set", help="persist choices, e.g. set cyberrealistic_pony=quant juggernaut=full")
    s.add_argument("pairs", nargs="*")
    s.add_argument("--default", choices=client.VARIANT_CHOICES, default=None)
    args = p.parse_args()

    if args.cmd == "convert":
        for key in (client.QUANT_MODEL_KEYS if args.all else [args.model]):
            src, dst = _paths(key)
            if args.all and os.path.exists(dst) and not args.force:
                print(f"{key}: already converted, skipping")
                continue
            convert(key, force=args.force, device=args.device, verify_after=not args.no_verify)
    elif args.cmd == "verify":
        verify(args.model)
    elif args.cmd == "status":
        status()
    elif args.cmd == "set":
        if not args.pairs and not args.default:
            raise SystemExit("nothing to set")
        set_choices(args.pairs, default=args.default)


if __name__ == "__main__":
    main()
