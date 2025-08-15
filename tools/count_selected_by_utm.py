# -*- coding: utf-8 -*-
"""
Count tiles per UTM zone from a splits file.

Usage:
  python tools/count_selected_by_utm.py \
    --splits data/splits/splits_verified_YYYYMMDD_HHMMSS.json \
    --set selected \
    [--update]  # optional: write the counts back into the splits JSON

For `--set`, you can use: selected, train, val, or test.
"""

import os, json, argparse
from collections import Counter, OrderedDict

def utm_zone_from_key(key: str) -> str:
    # keys look like "32638_99996" → "32638"
    if not key: return "unknown"
    p = key.split("_", 1)
    return p[0] if len(p) == 2 else "unknown"

def get_keys(payload: dict, set_name: str):
    if set_name == "selected":
        return list(payload.get("selected_tile_keys", []))
    return list(payload.get(set_name, []))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, help="path to selection/verified/final splits JSON")
    ap.add_argument("--set", default="selected", choices=["selected","train","val","test"],
                    help="which set to count; default: selected")
    ap.add_argument("--update", action="store_true",
                    help="if set, write counts back into the splits JSON under '<set>_utm_counts'")
    args = ap.parse_args()

    with open(args.splits, "r") as f:
        data = json.load(f)

    keys = get_keys(data, args.set)
    if not keys:
        raise SystemExit(f"[ERR] No keys found for set '{args.set}' in {args.splits}")

    counts = Counter(utm_zone_from_key(k) for k in keys)
    # stable, desc by count then zone
    counts_sorted = OrderedDict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    total = sum(counts_sorted.values())
    zones = len(counts_sorted)

    print(f"[OK] {args.set} set: {total} tiles across {zones} UTM zones\n")
    width = max(6, max(len(z) for z in counts_sorted))
    print(f"{'UTM'.ljust(width)}  count")
    print(f"{'-'*width}  -----")
    for z, c in counts_sorted.items():
        print(f"{z.ljust(width)}  {c}")

    if args.update:
        field = f"{args.set}_utm_counts"
        data[field] = counts_sorted  # OrderedDict serializes as JSON object
        with open(args.splits, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[UPDATE] Wrote counts into {args.splits} under '{field}'")

if __name__ == "__main__":
    main()
