import argparse
import json
import logging
import multiprocessing
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple, Set

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import tqdm


CATALOGUE_CACHE = "polyhaven_textures_catalogue.json"
FILES_API = "https://api.polyhaven.com/files/{asset_id}"
ASSETS_API = "https://api.polyhaven.com/assets?t=textures"


def save_json(obj, path: pathlib.Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: pathlib.Path) -> Optional[dict]:
  if path.exists():
    with path.open("r", encoding="utf-8") as f:
      return json.load(f)
  return None


def collect_list_of_available_assets(catalogue_path: str = CATALOGUE_CACHE) -> Dict[str, dict]:
  cache = pathlib.Path(catalogue_path)
  cached = load_json(cache)
  if cached:
    return cached
  resp = requests.get(ASSETS_API, timeout=30)
  resp.raise_for_status()
  data = resp.json()  # mapping id -> metadata
  save_json(data, cache)
  return data


def fetch_files_index(asset_id: str) -> dict:
  r = requests.get(FILES_API.format(asset_id=asset_id), timeout=30)
  r.raise_for_status()
  return r.json()


def preferred_formats_for_map(map_name: str) -> List[str]:
  # Prefer jpg for Diffuse (small, sRGB); for linear/non-color maps prefer exr, then png
  if map_name.lower() in ["diffuse", "albedo", "basecolor", "base color", "color"]:
    return ["jpg", "png", "exr"]
  return ["exr", "png", "jpg"]


def _iter_includes_for_resolution(files_index: dict, resolution: str):
  """Yield (include_path, include_meta) from blend includes for the given resolution.

  Supports both shapes:
  1) files_index[resolution]["blend"]["include"]
  2) files_index["blend"][resolution]["blend"]["include"]
  """
  # Shape 1
  try:
    inc = files_index.get(resolution, {}).get("blend", {}).get("include", {})
    if isinstance(inc, dict):
      for k, v in inc.items():
        yield k, v
  except Exception:
    pass
  # Shape 2
  try:
    inc = files_index.get("blend", {}).get(resolution, {}).get("blend", {}).get("include", {})
    if isinstance(inc, dict):
      for k, v in inc.items():
        yield k, v
  except Exception:
    pass


def _pick_from_includes(files_index: dict, map_key: str, resolution: str) -> Optional[Tuple[str, dict]]:
  """Pick a URL for map_key by scanning .blend includes for the chosen resolution."""
  key = map_key.lower()
  if key in ["diffuse", "albedo", "basecolor", "base color", "color"]:
    patterns = ["_diff_", "_albedo_", "_basecolor_", "_col_"]
  elif key in ["normalgl", "normal", "nor_gl"]:
    patterns = ["_nor_gl_", "_normalgl_", "_normal_"]
  elif key in ["roughness", "rough"]:
    patterns = ["_rough_"]
  elif key in ["displacement", "disp", "height"]:
    patterns = ["_disp_", "_height_", "_displ_"]
  else:
    patterns = []

  res_tag = f"_{resolution.lower()}"
  candidates: List[Tuple[str, dict]] = []
  for inc_path, meta in _iter_includes_for_resolution(files_index, resolution):
    fname = os.path.basename(inc_path).lower()
    if not fname:
      continue
    if res_tag not in fname:
      continue
    if any(p in fname for p in patterns):
      url = meta.get("url")
      if url:
        candidates.append((url, meta))

  if not candidates:
    return None

  prefs = preferred_formats_for_map(map_key)
  def ext_rank(u: str) -> int:
    ext = os.path.splitext(urllib.parse.urlparse(u).path)[1].lstrip(".").lower()
    try:
      return prefs.index(ext)
    except ValueError:
      return len(prefs)

  candidates.sort(key=lambda item: ext_rank(item[0]))
  return candidates[0]


def pick_url(files_index: dict, map_key: str, resolution: str) -> Optional[Tuple[str, dict]]:
  # Attempt direct shape: files_index[MapName][resolution][ext]
  try:
    if map_key in files_index:
      res_dict = files_index.get(map_key, {}).get(resolution)
      if isinstance(res_dict, dict):
        for ext in preferred_formats_for_map(map_key):
          meta = res_dict.get(ext)
          if isinstance(meta, dict) and "url" in meta:
            return meta["url"], meta
        for _ext, meta in res_dict.items():
          if isinstance(meta, dict) and "url" in meta:
            return meta["url"], meta
  except Exception:
    pass

  # Fallback to scanning .blend includes
  picked = _pick_from_includes(files_index, map_key, resolution)
  if picked is None:
    logging.debug(f"Map '{map_key}' not found for resolution '{resolution}'.")
  return picked


def normalize_map_keys(requested: List[str]) -> List[str]:
  # Accept common user inputs and map to API keys
  alias = {
    "diff": "Diffuse",
    "diffuse": "Diffuse",
    "albedo": "Diffuse",
    "basecolor": "Diffuse",
    "base_color": "Diffuse",
    "color": "Diffuse",
    "nor": "NormalGL",
    "normal": "NormalGL",
    "normalgl": "NormalGL",
    "nor_gl": "NormalGL",
    "normal_gl": "NormalGL",
    "rough": "Roughness",
    "roughness": "Roughness",
    "disp": "Displacement",
    "displacement": "Displacement",
    "height": "Displacement",
  }
  out = []
  for m in requested:
    key = alias.get(m.lower(), m)
    out.append(key)
  # Deduplicate but keep order
  seen = set()
  uniq = []
  for k in out:
    if k not in seen:
      uniq.append(k)
      seen.add(k)
  return uniq


def download_file(
  url: str,
  target_path: pathlib.Path,
  *,
  retries: int = 3,
  connect_timeout: int = 10,
  read_timeout: int = 30,
  max_file_seconds: Optional[int] = 600,
) -> bool:
  target_path.parent.mkdir(parents=True, exist_ok=True)
  if target_path.exists() and target_path.stat().st_size > 0:
    return True
  attempt = 0
  while True:
    attempt += 1
    start = time.time()
    try:
      with requests.get(
        url,
        stream=True,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=(connect_timeout, read_timeout),
      ) as r:
        r.raise_for_status()
        bytes_written = 0
        with target_path.open("wb") as f:
          for chunk in r.iter_content(chunk_size=1024 * 512):
            if chunk:
              f.write(chunk)
              bytes_written += len(chunk)
            # Deadline check to avoid infinite slow-drip
            if max_file_seconds is not None and (time.time() - start) > max_file_seconds:
              raise requests.ReadTimeout(f"Exceeded max_file_seconds={max_file_seconds}s for {url}")
      # Sanity check: non-empty file
      if target_path.stat().st_size == 0:
        raise IOError(f"Downloaded empty file for {url}")
      return True
    except (requests.RequestException, IOError) as e:
      if attempt > retries:
        logging.warning(f"Failed to download {url} after {retries} retries: {e}")
        # Clean up partial file
        try:
          if target_path.exists():
            target_path.unlink()
        except Exception:
          pass
        return False
      backoff = min(10 * attempt, 60)
      logging.info(f"Retry {attempt}/{retries} for {url} after error: {e}. Backing off {backoff}s...")
      time.sleep(backoff)


def _download_worker(
  args: Tuple[str, str, int, int, int, Optional[int]]
) -> Optional[str]:
  """Worker returns local path string on success, else None."""
  url, local_path_str, retries, connect_timeout, read_timeout, max_file_seconds = args
  ok = download_file(
    url,
    pathlib.Path(local_path_str),
    retries=retries,
    connect_timeout=connect_timeout,
    read_timeout=read_timeout,
    max_file_seconds=max_file_seconds,
  )
  return local_path_str if ok else None


def build_and_download(
  assets_meta: Dict[str, dict],
  download_dir: pathlib.Path,
  resolution: str = "4k",
  maps: Optional[List[str]] = None,
  limit: Optional[int] = None,
  processes: int = 8,
  *,
  retries: int = 3,
  connect_timeout: int = 10,
  read_timeout: int = 30,
  max_file_seconds: Optional[int] = 600,
) -> dict:
  maps = normalize_map_keys(maps or ["Diffuse", "NormalGL", "Roughness", "Displacement"])

  # Prepare a list of download tasks by peeking the files API per asset
  # Build plan
  tasks: List[Tuple[str, str]] = []  # (url, local_path_str)
  planned: Dict[str, Dict[str, dict]] = {}  # asset_id -> map_key -> info

  count = 0
  for asset_id, meta in assets_meta.items():
    if limit is not None and count >= limit:
      break
    try:
      files_index = fetch_files_index(asset_id)
    except Exception as e:
      logging.warning(f"Skip {asset_id}: failed to fetch files index: {e}")
      continue

    downloaded_info = {}
    for map_key in maps:
      picked = pick_url(files_index, map_key, resolution)
      if not picked:
        continue
      url, info = picked
      filename = pathlib.Path(urllib.parse.urlparse(url).path).name
      local_path = download_dir / asset_id / filename
      tasks.append((url, str(local_path)))
      downloaded_info[map_key] = {
        "url": url,
        "file": str(local_path.relative_to(download_dir)),
        "md5": info.get("md5"),
        "size": info.get("size"),
      }

    if downloaded_info:
      planned[asset_id] = downloaded_info
      count += 1

  # Parallel download (threaded for IO-bound tasks to avoid multiprocessing hangs)
  success_paths: Set[str] = set()
  if len(tasks) == 0:
    logging.info("No files to download.")
  else:
    with tqdm.tqdm(total=len(tasks)) as pbar:
      # Prepare arguments including timeouts and retries
      task_args = [
        (url, local_path_str, retries, connect_timeout, read_timeout, max_file_seconds)
        for (url, local_path_str) in tasks
      ]
      with ThreadPoolExecutor(max_workers=max(1, processes)) as executor:
        future_to_path = {executor.submit(_download_worker, ta): ta[1] for ta in task_args}
        for fut in as_completed(future_to_path):
          local_path_str = None
          try:
            local_path_str = fut.result()
          except Exception as e:
            logging.warning(f"Download task raised exception: {e}")
          if local_path_str:
            try:
              rel = str(pathlib.Path(local_path_str).resolve().relative_to(download_dir.resolve()))
            except Exception:
              rel = os.path.basename(local_path_str)
            success_paths.add(rel)
          pbar.update(1)

  # Build manifest only for successfully downloaded files
  manifest_assets: Dict[str, dict] = {}
  for asset_id, maps_info in planned.items():
    downloaded_only = {
      k: v for k, v in maps_info.items() if v.get("file") in success_paths
    }
    if not downloaded_only:
      continue
    meta = assets_meta.get(asset_id, {})
    manifest_assets[asset_id] = {
      "name": meta.get("name", asset_id),
      "authors": list(meta.get("authors", {}).keys()) if isinstance(meta.get("authors"), dict) else meta.get("authors"),
      "categories": meta.get("categories"),
      "tags": meta.get("tags"),
      "max_resolution": meta.get("max_resolution"),
      "thumbnail_url": meta.get("thumbnail_url"),
      "downloaded": downloaded_only,
    }

  return manifest_assets


def main():
  parser = argparse.ArgumentParser(description="Download Poly Haven textures by maps and resolution, and write a manifest JSON.")
  parser.add_argument("--download_dir", type=str, default="polyhaven_textures", help="Directory to save textures")
  parser.add_argument("--resolution", type=str, default="4k", help="Resolution to download (e.g., 1k, 2k, 4k, 8k, 16k)")
  parser.add_argument(
    "--maps",
    type=str,
    default="Diffuse,NormalGL,Roughness,Displacement",
    help="Comma-separated list of maps to download (e.g., Diffuse,NormalGL,Roughness,Displacement). Aliases: diff, nor_gl, rough, disp, height, albedo."
  )
  parser.add_argument("--limit", type=int, default=None, help="Limit number of assets (for testing)")
  parser.add_argument("--catalogue_cache", type=str, default=CATALOGUE_CACHE, help="Path to cache the assets catalogue JSON")
  parser.add_argument("--manifest", type=str, default="polyhaven_textures_manifest.json", help="Output manifest JSON path")
  parser.add_argument("--processes", type=int, default=min(8, os.cpu_count() or 8), help="Max concurrent downloads (threads)")
  parser.add_argument("--retries", type=int, default=3, help="Retries per file on failure")
  parser.add_argument("--connect_timeout", type=int, default=10, help="Connect timeout seconds")
  parser.add_argument("--read_timeout", type=int, default=30, help="Read timeout seconds per chunk")
  parser.add_argument("--max_file_seconds", type=int, default=600, help="Max total seconds allowed per file (abort if exceeded)")

  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

  download_dir = pathlib.Path(args.download_dir)
  maps = [m.strip() for m in args.maps.split(",") if m.strip()]

  logging.info("Fetching textures catalogue...")
  assets_meta = collect_list_of_available_assets(args.catalogue_cache)
  logging.info(f"Found {len(assets_meta)} assets in catalogue.")

  logging.info("Building download plan and downloading files...")
  manifest_assets = build_and_download(
    assets_meta=assets_meta,
    download_dir=download_dir,
    resolution=args.resolution,
    maps=maps,
    limit=args.limit,
    processes=args.processes,
    retries=args.retries,
    connect_timeout=args.connect_timeout,
    read_timeout=args.read_timeout,
    max_file_seconds=args.max_file_seconds,
  )

  manifest = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    "download_dir": str(download_dir.resolve()),
    "resolution": args.resolution,
    "maps": normalize_map_keys(maps),
    "assets": manifest_assets,
  }
  save_json(manifest, pathlib.Path(args.manifest))
  logging.info(f"Wrote manifest to {args.manifest} (assets with any downloaded maps: {len(manifest_assets)})")


if __name__ == "__main__":
  main()