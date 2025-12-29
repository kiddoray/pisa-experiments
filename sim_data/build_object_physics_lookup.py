#!/usr/bin/env python3
"""Build a physics lookup (friction, restitution) for GSO (or similar) object assets.

Usage:
  python build_object_physics_lookup.py --manifest gs://kubric-public/assets/GSO/GSO.json \
      --out /mnt/solo/kiddoray/GSO_physics.json

If --out is omitted, the output path defaults to manifest basename with suffix '_physics.json'.

Heuristic strategy:
  Base friction/restitution derived from metadata.category. Then adjusted by tokens
  found in the description (e.g. leather, rubber, glass, metal, wood, boot, sneaker,
  sandal). Additional adjustments made for very large volume (> 0.002 m^3) objects
  (slightly higher friction) and for very small volume (< 6e-5 m^3) (slightly lower
  friction).

The output schema:
  {
    "source_manifest": <path_or_url>,
    "count": <int>,
    "generated_at": <iso8601>,
    "lookup": {
        <asset_id>: {
            "friction": <float>,
            "restitution": <float>,
            "category": <str|None>,
            "tokens": [<str>, ...],
            "volume": <float|None>,
            "mass": <float|None>,
            "basis": {
                "base_category": <str|None>,
                "base_friction": <float>,
                "base_restitution": <float>
            }
        }, ...
    }
  }

This file is intentionally standalone and does not import Blender/Kubric.
"""

from __future__ import annotations
import os, sys, json, re, argparse, datetime, math
from urllib.parse import urlparse
from urllib.request import urlopen

CATEGORY_BASE = {
    # category : (friction, restitution)
    "Shoe": (0.90, 0.06),
    "Boot": (0.92, 0.05),  # if ever appears directly
    "Toys": (0.60, 0.20),
    "Action Figures": (0.55, 0.22),
    "Stuffed Toys": (0.80, 0.25),
    "Legos": (0.55, 0.28),
    "Board Games": (0.50, 0.30),
    "Consumer Goods": (0.50, 0.28),
    "Media Cases": (0.48, 0.32),
    "Bottles and Cans and Cups": (0.40, 0.35),
    "None": (0.55, 0.25),  # uncategorized / generic
}

TOKEN_RULES = [
    # (regex, friction_mul_or_set, restitution_mul_or_set, mode)
    # mode: 'set' replaces, 'scale' multiplies then clamps
    (r"metal|steel|aluminum|brass", (0.45, 0.40, 'set')),
    (r"glass|ceramic|porcelain", (0.35, 0.05, 'set')),
    (r"rubber", (1.15, 0.90, 'scale')),  # friction *1.15 (cap 0.95), restitution *0.90
    (r"wood|timber|oak|pine", (0.65, 0.25, 'set')),
    (r"leather|suede", (0.70, 0.15, 'set')),
    (r"sandal|flipflop|flip|thong", (0.80, 0.15, 'set')),
    (r"sneaker|running|athletic|trainer", (0.90, 0.06, 'set')),
    (r"boot", (0.88, 0.07, 'set')),
    (r"towel|cloth|fabric", (0.85, 0.20, 'set')),
    (r"plastic|polymer|abs", (0.50, 0.30, 'set')),
]

def load_manifest(path: str):
    parsed = urlparse(path)
    if parsed.scheme in ("http", "https"):
        with urlopen(path) as resp:
            return json.loads(resp.read().decode('utf-8'))
    if parsed.scheme == "gs":
        # Try https mirror if gs:// provided
        https_equiv = f"https://storage.googleapis.com/{parsed.netloc}{parsed.path}"
        try:
            with urlopen(https_equiv) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch gs:// manifest via HTTPS mirror: {https_equiv}: {e}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def infer_physics(category: str | None, description: str | None, volume: float | None) -> dict:
    cat = category or "None"
    base_friction, base_restitution = CATEGORY_BASE.get(cat, CATEGORY_BASE["None"])
    text = f"{category or ''} {description or ''}".lower()
    chosen_friction = base_friction
    chosen_restitution = base_restitution
    tokens_applied = []
    for pattern, (f_val, r_val, mode) in TOKEN_RULES:
        if re.search(pattern, text):
            tokens_applied.append(pattern)
            if mode == 'set':
                chosen_friction = f_val
                chosen_restitution = r_val
            else:  # scale
                chosen_friction = clamp(chosen_friction * f_val, 0.05, 0.95)
                chosen_restitution = clamp(chosen_restitution * r_val, 0.0, 0.95)
    # Size/volume adjustments
    if volume is not None:
        try:
            if volume > 0.002:  # large/heavy, more stable -> slightly higher friction
                chosen_friction = clamp(chosen_friction + 0.05, 0.05, 0.95)
            elif volume < 6e-5:  # tiny -> a bit less friction (easier to slide)
                chosen_friction = clamp(chosen_friction - 0.05, 0.05, 0.95)
        except Exception:
            pass
    return {
        "friction": round(chosen_friction, 4),
        "restitution": round(chosen_restitution, 4),
        "category": category,
        "tokens": tokens_applied,
        "basis": {
            "base_category": category,
            "base_friction": base_friction,
            "base_restitution": base_restitution,
        },
    }

def build_lookup(manifest: dict) -> dict:
    assets = manifest.get("assets", {})
    lut = {}
    for aid, ainfo in assets.items():
        if not isinstance(ainfo, dict):
            continue
        meta = ainfo.get("metadata", {}) or {}
        category = meta.get("category")
        description = meta.get("description")
        volume = meta.get("volume")
        kwargs = ainfo.get("kwargs", {}) or {}
        mass = kwargs.get("mass")
        rec = infer_physics(category, description, volume)
        rec["volume"] = volume
        rec["mass"] = mass
        lut[aid] = rec
    return lut

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path or URL to GSO manifest JSON")
    ap.add_argument("--out", default="", help="Optional output path; defaults to manifest basename + _physics.json")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output file")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    lut = build_lookup(manifest)
    out_path = args.out
    if not out_path:
        base = os.path.splitext(os.path.basename(args.manifest))[0]
        out_path = os.path.join(os.getcwd(), base + "_physics.json")

    if os.path.exists(out_path) and not args.overwrite:
        print(f"Refusing to overwrite existing file: {out_path} (use --overwrite)", file=sys.stderr)
        sys.exit(1)

    payload = {
        "source_manifest": args.manifest,
        "count": len(lut),
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "lookup": lut,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote object physics lookup: {out_path} (assets={len(lut)})")

if __name__ == '__main__':
    main()
