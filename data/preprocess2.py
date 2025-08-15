# File: data/preprocess.py
import os
import glob
import re
import json
import argparse
import yaml
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from collections import defaultdict, Counter
from scipy.stats import entropy as scipy_entropy

# ========== Utilities ==========

def extract_zone_and_index(path):
    basename = os.path.basename(path)
    # keep original UTM pattern; if your files include 3035 tags you can extend here
    zone_match = re.search(r'_(326\d{2})_', basename)
    index_match = re.findall(r"(\d+)(?=\.tif$)", basename)
    if zone_match and index_match:
        return zone_match.group(1), index_match[-1]
    return None, None

def center_crop(array, target_size=(512, 512)):
    if array.ndim == 3:
        _, h, w = array.shape
        th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return array[:, i:i+th, j:j+tw]
    elif array.ndim == 2:
        h, w = array.shape
        th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return array[i:i+th, j:j+tw]
    return array

def compute_histogram_and_entropy(label_array, num_classes):
    """
    label_array: uint8/uint16 2D array of labels (already in model ID space if you remap upstream)
    Returns integer histogram (length=num_classes) and a simple entropy of that histogram.
    """
    hist = np.bincount(label_array.flatten(), minlength=num_classes)
    hist = hist.astype(np.int64)
    # Add small epsilon to avoid log(0) if needed
    ent = scipy_entropy(hist + 1e-6, base=2)
    return hist.tolist(), float(ent)

def process_tile(path, num_classes, label_band):
    try:
        with rasterio.open(path) as src:
            label = src.read(label_band)
            label = label.astype(np.uint8)
            hist, ent = compute_histogram_and_entropy(label, num_classes)
        zone, index = extract_zone_and_index(path)
        return {
            "path": path,
            "zone": zone,
            "index": index,
            "label_hist": hist,
            "entropy": ent
        }
    except Exception as e:
        print(f"[WARN] Failed to process {path}: {e}")
        return None

# ========== Adaptive per-zone entropy selection helpers ==========

def _cfg_get(cfg, key, default):
    """
    Read from cfg['preprocess'][key] if present, else from cfg[key], else default.
    Keeps current configs working while allowing a preprocess: {} block.
    """
    return cfg.get("preprocess", {}).get(key, cfg.get(key, default))

def _entropy_edges(vals, bins):
    if bins <= 1 or len(vals) == 0:
        return []
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(vals, qs)
    # nudge to ensure strictly increasing edges
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-8
    return edges

def _bg_ratio_from_hist(hist):
    s = hist.sum()
    return (hist[0] / s) if s > 0 else 1.0

# ========== Main ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage", choices=["scan", "filter", "all"], default="all")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--chunksize", type=int, default=8)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    input_dir = cfg["input_dir"]
    num_classes = cfg["num_classes"]
    label_band = cfg.get("label_band", 1)

    # Filters (backward-compatible defaults)
    bg_threshold      = _cfg_get(cfg, "background_threshold", 0.95)
    entropy_bins      = int(_cfg_get(cfg, "entropy_bins", 2))              # 2=low/high; 4=quartiles
    per_zone_quota    = _cfg_get(cfg, "per_zone_quota", None)              # e.g., 500; None=keep all passing per zone
    target_low_frac   = float(_cfg_get(cfg, "target_low_frac", 0.5))       # only used when bins=2
    low_floor         = float(_cfg_get(cfg, "low_floor", 0.30))            # hard min low-entropy share per zone
    low_cap           = float(_cfg_get(cfg, "low_cap", 0.80))              # hard max low-entropy share per zone
    rare_class_ids    = list(_cfg_get(cfg, "rare_class_ids", []))          # optional nudge
    rare_class_bonus  = float(_cfg_get(cfg, "rare_class_bonus", 0.15))     # +15% weight during backfill
    global_cap        = _cfg_get(cfg, "global_cap", None)                  # e.g., 20000; None=no cap

    # Persistent histogram cache path
    hist_dir = os.path.join("data", "hist_cache")
    os.makedirs(hist_dir, exist_ok=True)
    hist_json = os.path.join(hist_dir, "tile_histograms.jsonl")

    # ===== Stage: SCAN =====
    if args.stage in ("scan", "all"):
        tif_paths = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
        print(f"[INFO] Found {len(tif_paths)} tiles")
        if len(tif_paths) == 0:
            raise SystemExit("[ERROR] No .tif files found in input_dir")

        # Write/overwrite cache (keeps behavior obvious; you can skip if file exists)
        with open(hist_json, "w") as fp:
            work = [(p, num_classes, label_band) for p in tif_paths]
            with Pool(processes=args.workers) as pool:
                for result in tqdm(pool.imap_unordered(
                        lambda x: process_tile(*x), work, chunksize=args.chunksize),
                        total=len(work), desc="Scanning"):
                    if result:
                        fp.write(json.dumps(result) + "\n")
        print(f"[INFO] Saved histograms to {hist_json}")

    # ===== Stage: FILTER (background + adaptive per-zone entropy) =====
    if args.stage in ("filter", "all"):
        if not os.path.exists(hist_json):
            raise SystemExit(f"[ERROR] Missing histogram cache: {hist_json}. Run with --stage scan first.")

        # Load records
        records = []
        with open(hist_json, "r") as fp:
            for line in fp:
                try:
                    rec = json.loads(line)
                    # ensure label_hist is np.array for quick math
                    rec["label_hist"] = np.array(rec["label_hist"], dtype=np.int64)
                    records.append(rec)
                except Exception:
                    pass

        if len(records) == 0:
            raise SystemExit("[ERROR] No histogram records loaded; cache is empty or corrupt.")

        # Background filter first
        candidates = []
        for r in records:
            bg_ratio = _bg_ratio_from_hist(r["label_hist"])
            if bg_ratio <= bg_threshold:
                candidates.append(r)

        print(f"[INFO] {len(candidates)} / {len(records)} tiles passed background threshold (≤ {bg_threshold:.2f})")

        # Group by zone
        by_zone = defaultdict(list)
        for r in candidates:
            z = r.get("zone") or "UNKNOWN"
            by_zone[z].append(r)

        # Adaptive per-zone entropy balancing
        selected_paths = []
        per_zone_stats = {}

        for z, items in by_zone.items():
            if not items:
                continue

            ents = np.array([float(it.get("entropy", 0.0)) for it in items], dtype=float)
            edges = _entropy_edges(ents, entropy_bins)

            # Assign entropy bin per tile (0..B-1); bin 0 = lowest entropy
            if entropy_bins > 1:
                bin_id = np.minimum(entropy_bins - 1, np.searchsorted(edges, ents, side="right") - 1)
            else:
                bin_id = np.zeros(len(items), dtype=int)

            # Build bins: store indices sorted by entropy desc inside each bin
            bins = []
            for b in range(entropy_bins):
                idx = np.where(bin_id == b)[0]
                # sort each bin high->low to prefer more informative tiles within the bin
                sub = sorted(idx, key=lambda i: items[i].get("entropy", 0.0), reverse=True)
                bins.append(sub)

            N_zone = sum(len(bi) for bi in bins)
            if N_zone == 0:
                continue

            # Quota for the zone
            Qz = int(per_zone_quota) if per_zone_quota is not None else N_zone
            Qz = min(Qz, N_zone)  # can't take more than available

            # Ideal per-bin target counts
            target_counts = [0] * entropy_bins
            if entropy_bins == 2:
                target_counts[0] = int(round(Qz * target_low_frac))
                target_counts[1] = Qz - target_counts[0]
            else:
                base = Qz // entropy_bins
                target_counts = [base] * entropy_bins
                target_counts[0] += Qz - base * entropy_bins  # put remainder in lowest bin; backfill will adjust

            # Clip targets by availability
            take = [min(target_counts[b], len(bins[b])) for b in range(entropy_bins)]
            leftover = Qz - sum(take)

            # Backfill leftover into bins with capacity
            if leftover > 0:
                capacity = np.array([max(0, len(bins[b]) - take[b]) for b in range(entropy_bins)], dtype=int)

                # Rare-class nudging (optional)
                bonus = np.zeros(entropy_bins, dtype=float)
                if len(rare_class_ids) > 0:
                    rc = set(int(x) for x in rare_class_ids)
                    for b in range(entropy_bins):
                        has_rare = False
                        for i in bins[b]:
                            hist = items[i]["label_hist"]
                            # if any rare class has presence in this tile
                            if any((cid < len(hist) and hist[cid] > 0) for cid in rc):
                                has_rare = True
                                break
                        if has_rare:
                            bonus[b] = rare_class_bonus

                weights = capacity.astype(float) * (1.0 + bonus)
                if weights.sum() > 0:
                    # proportional shares (largest remainder)
                    shares_f = (weights / weights.sum()) * leftover
                    shares = shares_f.astype(int)
                    rem = leftover - shares.sum()
                    order = np.argsort(-(shares_f - shares))
                    for k in order[:rem]:
                        shares[k] += 1
                    for b in range(entropy_bins):
                        take[b] += min(int(shares[b]), int(capacity[b]))

            # Enforce low-entropy floor/cap for B=2
            if entropy_bins == 2 and Qz > 0:
                low_share = take[0] / Qz
                # raise to floor if too low
                if low_share < low_floor:
                    need = int(np.ceil(Qz * low_floor)) - take[0]
                    move = min(need, max(0, len(bins[1]) - take[1]))
                    take[0] += max(0, move)
                    take[1] -= max(0, move)
                # reduce to cap if too high
                low_share = take[0] / Qz
                if low_share > low_cap:
                    give = int(np.floor(Qz * (low_share - low_cap)))
                    move = min(give, take[0])
                    take[0] -= max(0, move)
                    take[1] += max(0, move)

            # Final selection for this zone
            sel_idx = []
            for b in range(entropy_bins):
                sel_idx.extend(bins[b][:take[b]])

            # Record
            selected_paths.extend([items[i]["path"] for i in sel_idx])
            per_zone_stats[z] = {
                "available": N_zone,
                "selected": len(sel_idx),
                "take_per_bin": take,
                "bins_available": [len(bi) for bi in bins]
            }

        # Optional: global cap (keep most informative globally)
        if global_cap is not None and len(selected_paths) > int(global_cap):
            # Build a quick lookup: path -> entropy
            ent_map = {r["path"]: float(r.get("entropy", 0.0)) for r in candidates}
            selected_paths.sort(key=lambda p: ent_map.get(p, 0.0), reverse=True)
            selected_paths = selected_paths[:int(global_cap)]

        selected_set = set(selected_paths)

        # Build final selected records (preserve the structure you already use downstream)
        final = [r for r in candidates if r["path"] in selected_set]

        # ===== Save outputs =====
        hist_dir = os.path.join("data", "hist_cache")
        os.makedirs(hist_dir, exist_ok=True)
        filtered_json = os.path.join(hist_dir, "filtered_tiles.json")  # keep filename for backward compatibility
        stats_json = os.path.join(hist_dir, "filter_stats.json")

        with open(filtered_json, "w") as fp:
            # convert numpy arrays back to lists for JSON
            serializable = []
            for r in final:
                rr = dict(r)
                rr["label_hist"] = r["label_hist"].tolist()
                serializable.append(rr)
            json.dump(serializable, fp, indent=2)

        # Small stats dump (handy in logs)
        totals = {
            "total_scanned": len(records),
            "passed_background": len(candidates),
            "selected_final": len(final),
            "zones_seen": len(by_zone),
            "bg_threshold": bg_threshold,
            "entropy_bins": entropy_bins,
            "per_zone_quota": per_zone_quota,
            "target_low_frac": target_low_frac,
            "low_floor": low_floor,
            "low_cap": low_cap,
            "rare_class_ids": rare_class_ids,
            "rare_class_bonus": rare_class_bonus,
            "global_cap": global_cap,
            "per_zone": per_zone_stats
        }
        with open(stats_json, "w") as fp:
            json.dump(totals, fp, indent=2)

        # Console summary
        print(f"[INFO] Saved filtered tile list to {filtered_json}")
        print(f"[INFO] Selected {len(final)} tiles across {len(by_zone)} zones.")
        zline = ", ".join(f"{z}:{per_zone_stats.get(z, {}).get('selected', 0)}"
                          for z in sorted(per_zone_stats))
        print(f"[ZONES] {zline}")
