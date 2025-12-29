#!/usr/bin/env python3
"""
Build a physics lookup table (friction, restitution) for Poly Haven textures.

- Input: a manifest JSON produced by your downloader that includes, per asset:
  {
    "assets": {
      "asset_id": {
        "tags": [...],
        "categories": [...],
        "downloaded": { ... }
      },
      ...
    },
    "download_dir": "/mnt/solo/kiddoray/Texture_haven"
  }
- Output: a JSON written next to the manifest named <manifest>_physics.json containing
  {
    "source_manifest": <path>,
    "count": <int>,
    "lookup": { asset_id: { friction, restitution, basis, tags, categories } }
  }

Usage:
  python scripts/sim_data/build_texture_physics_lookup.py \
    --manifest /mnt/solo/kiddoray/Texture_haven.json

Optional:
  --out /path/to/output.json    # default: <manifest>_physics.json
  --overwrite                   # overwrite if file exists
"""
from __future__ import annotations
import argparse
import json
import os
from typing import List, Tuple


def _infer_physics_from_tags_categories(name: str | None, tags: list | None, categories: list | None):
    """Infer (friction, restitution) from tags/categories/asset name.

    Heuristics:
    - Base class mapping first match wins (asphalt, concrete, wood, metal, rubber, glass, tile, ice, snow, sand, grass, gravel, stone, fabric, plastic, marble, etc.)
    - Modifiers: wet/polished/glossy (reduce mu), rough/grunge/cracked (increase mu), mud (sticky), snow/ice (very low mu)
    - Clamps: mu in [0.02, 1.0], e in [0.0, 0.8]
    Returns (mu, e, basis_token).
    """
    def tokenize(s: str) -> list:
        if not s:
            return []
        s = s.replace('-', ' ').replace('_', ' ')
        return [t for t in s.lower().split() if t]

    tokens = set()
    for t in (tags or []):
        try:
            tokens.add(str(t).lower())
        except Exception:
            continue
    for c in (categories or []):
        try:
            tokens.add(str(c).lower())
        except Exception:
            continue
    for t in tokenize(name or ""):
        tokens.add(t)

    base = [
        ("ice", (0.03, 0.05)),
        ("icy", (0.03, 0.05)),
        ("snow", (0.05, 0.05)),
        ("glass", (0.20, 0.05)),
        ("asphalt", (0.85, 0.05)),
        ("road", (0.80, 0.05)),
        ("concrete", (0.80, 0.05)),
        ("cement", (0.78, 0.05)),
        ("stone", (0.75, 0.05)),
        ("rock", (0.75, 0.05)),
        ("brick", (0.75, 0.05)),
        ("gravel", (0.70, 0.10)),
        ("sand", (0.60, 0.05)),
        ("soil", (0.65, 0.10)),
        ("dirt", (0.70, 0.10)),
        ("grass", (0.70, 0.15)),
        ("moss", (0.75, 0.15)),
        ("wood", (0.60, 0.20)),
        ("tile", (0.50, 0.05)),
        ("ceramic", (0.50, 0.05)),
        ("marble", (0.45, 0.05)),
        ("metal", (0.40, 0.05)),
        ("steel", (0.42, 0.05)),
        ("aluminum", (0.38, 0.05)),
        ("plastic", (0.40, 0.20)),
        ("rubber", (1.00, 0.50)),
        ("fabric", (0.80, 0.30)),
        ("leather", (0.60, 0.25)),
        ("painted", (0.45, 0.10)),
        ("plaster", (0.65, 0.05)),
        ("terracotta", (0.55, 0.05)),
    ]

    mu, e, basis = 0.80, 0.05, "default"
    for key, (mu0, e0) in base:
        if key in tokens:
            mu, e, basis = mu0, e0, key
            break

    if any(k in tokens for k in ["wet", "polished", "glossy", "shiny"]):
        mu *= 0.6
        e = max(e, 0.05)
    if any(k in tokens for k in ["rough", "grunge", "broken", "cracked", "rugged", "matte"]):
        mu *= 1.15
    if "mud" in tokens:
        mu = max(mu, 0.85)
        e = min(e, 0.08)
    if any(k in tokens for k in ["snow", "ice", "icy"]):
        mu = min(mu, 0.07)
        e = min(e, 0.08)

    mu = float(max(0.02, min(1.00, mu)))
    e = float(max(0.0, min(0.80, e)))
    return mu, e, basis


def build_lookup(manifest_path: str, out_path: str) -> int:
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    assets = manifest.get("assets", {})
    lookup = {}
    for aid, ainfo in assets.items():
        try:
            tags = list(ainfo.get("tags", []) or [])
            cats = list(ainfo.get("categories", []) or [])
            mu, e, basis = _infer_physics_from_tags_categories(aid, tags, cats)
            lookup[aid] = {
                "friction": mu,
                "restitution": e,
                "basis": basis,
                "tags": tags,
                "categories": cats,
            }
        except Exception:
            continue

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "source_manifest": manifest_path,
            "count": len(lookup),
            "lookup": lookup,
        }, f, indent=2)
    return len(lookup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="/mnt/solo/kiddoray/Texture_haven.json",
                    help="Path to the Poly Haven downloader manifest JSON")
    ap.add_argument("--out", type=str, default=None,
                    help="Output JSON path for the physics lookup (default: <manifest>_physics.json)")
    ap.add_argument("--overwrite", action="store_true", default=False,
                    help="Overwrite existing output file if present")
    args = ap.parse_args()

    manifest_path = args.manifest
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if args.out:
        out_path = args.out
    else:
        root, ext = os.path.splitext(manifest_path)
        out_path = root + "_physics.json"

    if os.path.exists(out_path) and not args.overwrite:
        print(f"Output exists, skipping (use --overwrite to force): {out_path}")
        return

    count = build_lookup(manifest_path, out_path)
    print(f"Wrote texture physics lookup with {count} entries -> {out_path}")


if __name__ == "__main__":
    main()
