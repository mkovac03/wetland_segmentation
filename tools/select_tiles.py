# File: tools/select_tiles.py
# Config-driven tile selector with safe caching and hist_cache fallback.
#
# Behavior:
#   1) Read paths + thresholds from config YAML.
#   2) If JSONL cache exists (and no --force) -> use it.
#   3) Else rebuild JSONL from per-tile hist_cache JSONs (NO GeoTIFF reads).
#      Accepts forms like:
#        {"tile_id": "32634_184075", "histogram": {"0": 257584, "3": 2947, "10": 2398}}
#        {"tile_key": "...", "hist": {...}, "total": N}
#        {"0": 257584, "3": 2947, "10": 2398}  (filename encodes key)
#      IMPORTANT: No label remapping here — your histograms & labels are already final IDs.
#   4) Only if --force and caches missing -> scan labels to build JSONL.
#   5) Selection uses intersection of inputs/labels pairs & cache; water>50% is filtered
#      even in --mode simple.
#
# Run (config-only paths):
#   python tools/select_tiles.py --config configs/config.yaml --mode simple --workers -1

import os
import re
import json
import argparse
import yaml
from datetime import datetime
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import rasterio
from scipy.stats import entropy

# ---------------- Robust tile-key parsing ----------------
# Match ".../32634_184075.(tif|json|...)", works if embedded in longer names
RX_KEY_GENERIC = re.compile(r'(32[67]\d{2})_(\d+)')
RX_TIF_ID_AT_END = re.compile(r'(\d+)(?=\.tif$)', re.IGNORECASE)

def parse_tile_key_from_string(s: str):
    """Extract '326xx_<id>' from any filename or string."""
    base = os.path.basename(s)
    m = RX_KEY_GENERIC.search(base)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    # Fallback: if looks like "..._32634_184075.tif"
    zm = re.search(r'_(32[67]\d{2})_', base)
    if zm:
        zone = zm.group(1)
        im = RX_TIF_ID_AT_END.search(base)
        if zone and im:
            return f"{zone}_{im.group(1)}"
    return None

# ---------------- Indexing ----------------
def index_dir(root, recursive=True):
    out = defaultdict(list)
    if not root or not os.path.isdir(root):
        return out
    walker = os.walk(root) if recursive else [(root, [], os.listdir(root))]
    for dp, _, files in walker:
        for f in files:
            if f.lower().endswith(".tif"):
                key = parse_tile_key_from_string(f)
                if key:
                    out[key].append(os.path.join(dp, f))
    return out

def index_dirs(roots, recursive=True):
    merged = defaultdict(list)
    for r in roots:
        for k, v in index_dir(r, recursive=recursive).items():
            merged[k].extend(v)
    return merged

# ---------------- Config + thresholds ----------------
def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    sel = cfg.get("select", {})
    split = cfg.get("splitting", {})

    sel.setdefault("background_class", 0)
    sel.setdefault("water_class", 10)
    sel.setdefault("ignore_val", cfg.get("ignore_val", 255))
    sel.setdefault("num_classes", cfg.get("num_classes", 13))
    # Default wetland set (tune to your taxonomy)
    sel.setdefault("wetland_classes", [1,2,3,4,5,6,7,8,9,10])
    sel.setdefault("include_water_in_wetland_fraction", True)
    sel.setdefault("bg_threshold", split.get("background_threshold", 0.90))
    sel.setdefault("water_max_fraction", 0.50)  # enforce globally
    sel.setdefault("wetland_min_fraction", 0.01)
    sel.setdefault("wetland_max_fraction", 0.50)
    sel.setdefault("entropy_min", 0.0)
    cfg["select"] = sel

    # NOTE: We intentionally do NOT use label_remap here. Your preprocessing
    # already mapped labels/histograms to final IDs.

    return cfg

def resolve_paths(cfg):
    # Base layout with subfolders
    base = cfg.get("input_dir")
    inputs_dir_from_base = os.path.join(base, "inputs") if base else None
    labels_dir_from_base = os.path.join(base, "labels") if base else None
    cache_dir_from_base  = os.path.join(base, "hist_cache") if base else None

    # Optional explicit overrides
    inputs_dir = cfg.get("inputs_dir")
    labels_dir = cfg.get("labels_dir")
    hist_cache_dir = cfg.get("hist_cache_dir")

    # input_dirs list (additional imagery roots)
    extra_inputs = cfg.get("input_dirs", []) or []

    # Build final list of input roots (explicit > inferred)
    input_roots = []
    if inputs_dir and os.path.isdir(inputs_dir): input_roots.append(inputs_dir)
    if inputs_dir_from_base and os.path.isdir(inputs_dir_from_base): input_roots.append(inputs_dir_from_base)
    for p in extra_inputs:
        if os.path.isdir(p): input_roots.append(p)

    # Labels + cache
    labels_root = labels_dir if labels_dir else labels_dir_from_base
    hist_cache_root = hist_cache_dir if hist_cache_dir else cache_dir_from_base

    # JSONL canonical (next to labels)
    jsonl_path = os.path.join(labels_root if labels_root else "data/labels", "label_histograms.jsonl")

    # Split path (supports {now})
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    splits_path = cfg.get("splits_path", f"data/splits/splits_{now}.json").format(now=now)

    return {
        "input_roots": input_roots,
        "labels_root": labels_root,
        "hist_cache_root": hist_cache_root,
        "jsonl_path": jsonl_path,
        "splits_path": splits_path,
        "now": now
    }

# ---------------- Metrics ----------------
def metrics_from_hist(hist: dict, total: int, cfg):
    sel = cfg["select"]
    bg = sel["background_class"]
    water = sel["water_class"]
    wet_set = set(sel["wetland_classes"])
    include_water = sel["include_water_in_wetland_fraction"]

    def frac(k): return (hist.get(k, 0) / total) if total > 0 else 0.0
    bg_frac = frac(bg)
    water_frac = frac(water)

    wet_keys = set(wet_set)
    if not include_water and water in wet_keys:
        wet_keys.remove(water)
    wet_count = sum(hist.get(k, 0) for k in wet_keys)
    wet_frac = (wet_count / total) if total > 0 else 0.0

    if total > 0 and hist:
        probs = np.array(list(hist.values()), dtype=np.float64)
        probs = probs / probs.sum()
        ent = float(entropy(probs))
    else:
        ent = 0.0

    return {"bg_frac": bg_frac, "water_frac": water_frac, "wet_frac": wet_frac, "entropy": ent}

def tile_reason(metrics, cfg, mode):
    sel = cfg["select"]

    # Always enforce >50% water cap (or configured cap)
    if metrics["water_frac"] > sel["water_max_fraction"]:
        return "water_gt_threshold"

    if mode == "simple":
        # Accept if any non-background present (after water check)
        return None if metrics["bg_frac"] < 1.0 and not np.isclose(metrics["bg_frac"], 1.0) else "all_background"

    # filtered mode thresholds
    if metrics["bg_frac"] > sel["bg_threshold"]:
        return "background_gt_threshold"
    if metrics["wet_frac"] < sel["wetland_min_fraction"]:
        return "wetland_lt_min"
    if metrics["wet_frac"] > sel["wetland_max_fraction"]:
        return "wetland_gt_max"
    if metrics["entropy"] < sel["entropy_min"]:
        return "entropy_lt_min"
    return None

# ---------------- Histogram utilities ----------------
def normalize_histogram(src_hist: dict, cfg):
    """Treat cached histogram keys as FINAL class IDs (no remap). Drop ignore if present."""
    ignore_val = cfg["select"].get("ignore_val", 255)
    out = {}
    for k, v in (src_hist or {}).items():
        try:
            kk = int(k)
            vv = int(v)
        except Exception:
            continue
        if kk == ignore_val:
            continue
        out[kk] = out.get(kk, 0) + vv
    return out

def extract_tile_key_from_record(rec: dict, filename: str):
    # prefer explicit fields
    key = None
    if isinstance(rec, dict):
        key = rec.get("tile_id") or rec.get("tile_key")
        if not key:
            # sometimes a path may be stored
            for fld in ("label_path", "path", "file"):
                if fld in rec and rec[fld]:
                    key = parse_tile_key_from_string(str(rec[fld]))
                    if key: break
    if not key:
        key = parse_tile_key_from_string(filename)
    return key

# ---------------- Slow path (ONLY with --force) ----------------
def read_label_hist(label_path, cfg):
    """Read label GeoTIFF and count class histogram. No remapping (labels are final)."""
    ignore_val = cfg["select"]["ignore_val"]
    with rasterio.open(label_path) as src:
        arr = src.read(1)
        nodata = src.nodata
    arr = arr.astype(np.int64)
    if nodata is not None:
        arr[arr == nodata] = ignore_val
    valid = arr != ignore_val
    total = int(valid.sum())
    hist = {}
    if total > 0:
        vals, cnts = np.unique(arr[valid], return_counts=True)
        hist = {int(k): int(v) for k, v in zip(vals.tolist(), cnts.tolist())}
    return hist, total

def worker(job):
    key, label_path, cfg, mode = job
    try:
        hist, total = read_label_hist(label_path, cfg)
        if total == 0:
            return key, None, {"reason": "all_ignore_or_nodata"}
        m = metrics_from_hist(hist, total, cfg)
        r = tile_reason(m, cfg, mode)
        return key, {"hist": hist, "total": total, **m}, {"reason": r}
    except Exception as e:
        return key, None, {"reason": f"error:{str(e)}"}

# ---------------- JSONL cache I/O ----------------
def load_cached_jsonl(jsonl_path):
    cache = {}
    if not (jsonl_path and os.path.exists(jsonl_path)):
        return cache
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            k = rec.get("tile_key")
            if not k:
                # try to derive from any stored path
                for fld in ("label_path", "path", "file"):
                    if isinstance(rec, dict) and fld in rec:
                        k = parse_tile_key_from_string(rec[fld])
                        if k: break
            if not k:
                continue
            entry = {}
            if "hist" in rec and "total" in rec:
                try:
                    entry["hist"] = {int(a): int(b) for a, b in rec["hist"].items()}
                    entry["total"] = int(rec["total"])
                except Exception:
                    entry["hist"], entry["total"] = None, int(rec.get("total", 0))
            else:
                entry["hist"] = None
                entry["total"] = int(rec.get("total", 0))
            for m in ("bg_frac", "water_frac", "wet_frac", "entropy"):
                if m in rec:
                    try: entry[m] = float(rec[m])
                    except Exception: pass
            cache[k] = entry
    return cache

def rebuild_jsonl_from_hist_cache(hist_cache_dir, jsonl_path, cfg, recursive=True, show_progress=True):
    """
    Build JSONL from per-tile JSON files in hist_cache_dir (NO GeoTIFF reads).
    Accepts any of:
      A) {"tile_id": "...", "histogram": {cls: count, ...}}
      B) {"tile_key": "...", "hist": {...}, "total": N}
      C) Raw {"cls": count, ...} where filename encodes the key (e.g., 32634_184075.json)
    No remap is applied; histogram keys are final class IDs. Ignore bin is dropped.
    Shows a progress bar while converting.
    """
    if not hist_cache_dir or not os.path.isdir(hist_cache_dir):
        return 0

    # Gather file list (so we can show total in the progress bar)
    files = []
    walker = os.walk(hist_cache_dir) if recursive else [(hist_cache_dir, [], os.listdir(hist_cache_dir))]
    for dp, _, fns in walker:
        for f in fns:
            if f.lower().endswith(".json"):
                files.append(os.path.join(dp, f))

    total_files = len(files)
    if total_files == 0:
        return 0

    entries = 0
    malformed = 0
    empty_after_norm = 0
    tmp = jsonl_path + ".tmp"

    def normalize_histogram(src_hist: dict, cfg):
        ignore_val = cfg["select"].get("ignore_val", 255)
        out = {}
        for k, v in (src_hist or {}).items():
            try:
                kk = int(k); vv = int(v)
            except Exception:
                continue
            if kk == ignore_val:
                continue
            out[kk] = out.get(kk, 0) + vv
        return out

    def extract_tile_key_from_record(rec: dict, filename: str):
        key = None
        if isinstance(rec, dict):
            key = rec.get("tile_id") or rec.get("tile_key")
            if not key:
                for fld in ("label_path", "path", "file"):
                    if fld in rec and rec[fld]:
                        k2 = parse_tile_key_from_string(str(rec[fld]))
                        if k2:
                            key = k2
                            break
        if not key:
            key = parse_tile_key_from_string(filename)
        return key

    with open(tmp, "w", buffering=1 << 20) as out:
        for p in tqdm(files, total=total_files, disable=not show_progress,
                      desc="hist_cache → JSONL", unit="file", dynamic_ncols=True):
            f = os.path.basename(p)
            try:
                with open(p, "r") as j:
                    rec = json.load(j)
            except Exception:
                malformed += 1
                continue

            key = extract_tile_key_from_record(rec, f)
            if not key:
                malformed += 1
                continue

            # detect source histogram
            src_hist = None
            if isinstance(rec, dict):
                if "histogram" in rec and isinstance(rec["histogram"], dict):
                    src_hist = rec["histogram"]
                elif "hist" in rec and isinstance(rec["hist"], dict):
                    src_hist = rec["hist"]
                else:
                    if rec and all(isinstance(v, int) for v in rec.values()):
                        src_hist = rec  # raw dict

            if not src_hist:
                malformed += 1
                continue

            hist = normalize_histogram(src_hist, cfg)
            total = int(sum(hist.values()))

            if total == 0:
                empty_after_norm += 1
                out.write(json.dumps({
                    "tile_key": key,
                    "hist": {},
                    "total": 0,
                    "bg_frac": 1.0,
                    "water_frac": 0.0,
                    "wet_frac": 0.0,
                    "entropy": 0.0,
                    "method": "from_hist_cache"
                }) + "\n")
                entries += 1
            else:
                m = metrics_from_hist(hist, total, cfg)
                out.write(json.dumps({
                    "tile_key": key,
                    "hist": {int(k): int(v) for k, v in hist.items()},
                    "total": total,
                    **m,
                    "method": "from_hist_cache"
                }) + "\n")
                entries += 1

            # occasional flush so you see progress even if interrupted
            if entries % 5000 == 0:
                out.flush()

    if entries > 0:
        os.replace(tmp, jsonl_path)
    else:
        if os.path.exists(tmp):
            os.remove(tmp)

    print(f"[FALLBACK] hist_cache scan complete: files={total_files}, written={entries}, "
          f"malformed={malformed}, empty_after_norm={empty_after_norm}")
    return entries


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--mode", choices=["simple", "filtered"], default="filtered")
    ap.add_argument("--max_selected", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="Rescan labels if caches missing/empty.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)

    # Index inputs & labels
    inputs_idx = index_dirs(paths["input_roots"], recursive=True)
    labels_idx = index_dir(paths["labels_root"], recursive=True)

    keys = sorted(set(inputs_idx.keys()) & set(labels_idx.keys()))
    print(f"[INFO] inputs_roots: {paths['input_roots']}")
    print(f"[INFO] labels_root:  {paths['labels_root']}")
    print(f"[INFO] hist_cache:   {paths['hist_cache_root']}")
    print(f"[DEBUG] Indexed labels: {len(labels_idx)}, inputs: {len(inputs_idx)}, pairs: {len(keys)}")
    print(f"[INFO] Found {len(keys)} pairs")

    # Ensure output dirs
    os.makedirs(os.path.dirname(paths["splits_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(paths["jsonl_path"]), exist_ok=True)

    # 1) Use existing JSONL if present (and not forced)
    if os.path.exists(paths["jsonl_path"]) and not args.force:
        print(f"[CACHE] Using existing JSONL → {paths['jsonl_path']}")
        cache = load_cached_jsonl(paths["jsonl_path"])
    else:
        # 2) Rebuild JSONL from hist_cache (no rescanning)
        print(f"[FALLBACK] JSONL missing or --force; trying hist_cache → {paths['hist_cache_root']}")
        entries = rebuild_jsonl_from_hist_cache(paths["hist_cache_root"], paths["jsonl_path"], cfg, recursive=True)
        if entries > 0:
            print(f"[FALLBACK] Rebuilt JSONL from hist_cache with {entries} entries → {paths['jsonl_path']}")
            cache = load_cached_jsonl(paths["jsonl_path"])
        else:
            cache = {}
            print("[FALLBACK] hist_cache empty or unusable.")
            # 3) Rescan labels ONLY if --force is given
            if args.force:
                if len(keys) == 0:
                    print("[GUARD] No pairs found; aborting scan and writing empty selection.")
                else:
                    print("[SCAN] Computing histograms by reading labels (slow path).")
                    jobs = []
                    for k in keys:
                        label_path = sorted(labels_idx[k])[0]
                        jobs.append((k, label_path, cfg, args.mode))
                    W = max(1, cpu_count()-1) if args.workers == -1 else max(1, args.workers)
                    tmp = paths["jsonl_path"] + ".tmp"
                    reason_counter = Counter()
                    with open(tmp, "w") as out:
                        if W > 1:
                            with Pool(W) as pool:
                                for k, metrics, status in tqdm(pool.imap_unordered(worker, jobs, chunksize=64),
                                                               total=len(jobs), desc="Scanning tiles"):
                                    if metrics is not None:
                                        out.write(json.dumps({"tile_key": k, **metrics}) + "\n")
                                    else:
                                        reason_counter[status["reason"]] += 1
                        else:
                            for job in tqdm(jobs, desc="Scanning tiles"):
                                k, metrics, status = worker(job)
                                if metrics is not None:
                                    out.write(json.dumps({"tile_key": k, **metrics}) + "\n")
                                else:
                                    reason_counter[status["reason"]] += 1
                    os.replace(tmp, paths["jsonl_path"])
                    print(f"[INFO] Wrote histograms → {paths['jsonl_path']}")
                    cache = load_cached_jsonl(paths["jsonl_path"])
            else:
                print("[ABORT] Not rescanning (no --force). Proceeding with empty cache.")

    # -------- Selection from cache (no rescanning) --------
    accept = []
    reason_counter = Counter()
    intersect_keys = [k for k in keys if k in cache]
    missing = len(keys) - len(intersect_keys)
    if missing > 0:
        print(f"[WARN] {missing} paired tiles not present in cache; excluded from selection.")

    for k in tqdm(intersect_keys, desc="Selecting from cache"):
        rec = cache[k]
        if all(x in rec for x in ("bg_frac", "water_frac", "wet_frac", "entropy")):
            m = {x: rec[x] for x in ("bg_frac", "water_frac", "wet_frac", "entropy")}
        elif rec.get("hist") is not None and rec.get("total", 0) > 0:
            m = metrics_from_hist(rec["hist"], rec["total"], cfg)
        else:
            reason_counter["no_cached_metrics"] += 1
            continue
        r = tile_reason(m, cfg, args.mode)
        if r is None:
            accept.append(k)
        else:
            reason_counter[r] += 1

    usable = len(accept)
    if args.max_selected is not None:
        accept = accept[:args.max_selected]

    with open(paths["splits_path"], "w") as f:
        json.dump({"mode": args.mode, "selected_tile_keys": accept}, f, indent=2)
    print(f"[OK] Wrote selection → {paths['splits_path']}")
    print(f"     Selected: {len(accept)} / usable: {usable} / pairs: {len(keys)}")

    if reason_counter:
        print("\n[DIAG] Exclusion breakdown:")
        for r, c in reason_counter.most_common():
            print(f"  {r:>26} : {c}")

    sel = cfg["select"]
    print("\n[INFO] Effective thresholds (filtered mode):")
    print(f"  bg_threshold={sel['bg_threshold']}, water_max_fraction={sel['water_max_fraction']}, "
          f"wetland_fraction=[{sel['wetland_min_fraction']}, {sel['wetland_max_fraction']}], "
          f"entropy_min={sel['entropy_min']}, include_water_in_wetland_fraction={sel['include_water_in_wetland_fraction']}")

if __name__ == "__main__":
    main()
