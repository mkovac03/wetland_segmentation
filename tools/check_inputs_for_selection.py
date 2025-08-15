# -*- coding: utf-8 -*-
# STRICT verifier: uses ONLY <input_dir>/inputs and <input_dir>/labels.
# Key format derived from filename/path: last UTM token 32[67]xx + last numeric token (tile id).
#
# Run:
#   python tools/check_inputs_for_selection.py \
#       --config configs/config.yaml \
#       --splits data/splits/splits_YYYYMMDD_HHMMSS.json \
#       --out data/splits/splits_verified_YYYYMMDD_HHMMSS.json

import os, re, json, argparse, yaml
from datetime import datetime

RX_NOW = re.compile(r'(\d{8}_\d{6})')

def extract_key_from_path(path: str):
    """
    From names like:
      google_embed_..._32638_bands_00_63_99996.tif  -> 32638_99996
      ext_wetland_..._32638_99996.tif               -> 32638_99996
    Logic:
      - zone  = last token matching 32[67]\d{2}
      - tile  = last numeric token in stem (if equals zone, use previous numeric)
    """
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    nums = re.findall(r'\d+', stem)                 # all numeric tokens
    if not nums: return None
    zones = [n for n in nums if re.fullmatch(r'32[67]\d{2}', n)]
    if not zones: return None
    zone = zones[-1]
    # prefer the last "_<digits>" group in the stem as tile id
    tail_ids = re.findall(r'_(\d+)', stem)
    tile = tail_ids[-1] if tail_ids else nums[-1]
    if tile == zone and len(nums) >= 2:
        tile = nums[-2]
    if tile == zone:  # still same? give up
        return None
    return f"{zone}_{tile}"

def extract_now_from_path(path: str):
    m = RX_NOW.search(os.path.basename(path))
    return m.group(1) if m else datetime.now().strftime("%Y%m%d_%H%M%S")

def load_cfg(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def index_keys(root, exts):
    keys = set()
    if not root or not os.path.isdir(root):
        return keys
    exts = tuple(e.lower() for e in exts)
    for dp, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                full = os.path.join(dp, f)
                k = extract_key_from_path(full)
                if k:
                    keys.add(k)
    return keys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exts", default="tif,tiff,vrt,npy", help="comma-separated input extensions")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    base = cfg.get("input_dir")
    if not base:
        raise SystemExit("[ERR] config.input_dir is not set.")
    inputs_root = os.path.join(base, "inputs")
    labels_root = os.path.join(base, "labels")

    with open(args.splits, "r") as f:
        sel = json.load(f)
    selected = list(sel.get("selected_tile_keys", []))
    mode = sel.get("mode", "simple")
    print(f"[INFO] Selected tiles: {len(selected)}")
    print(f"[INFO] inputs_root: {inputs_root}")
    print(f"[INFO] labels_root: {labels_root}")

    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    present_inputs = index_keys(inputs_root, exts)
    present_labels = index_keys(labels_root, ("tif","tiff","vrt"))

    print(f"[INFO] Indexed: inputs={len(present_inputs)} keys, labels={len(present_labels)} keys")

    selected_set = set(selected)
    have_both = [k for k in selected if (k in present_inputs and k in present_labels)]

    # Small debug samples if things are missing
    dropped = len(selected) - len(have_both)
    if dropped > 0:
        miss_in = sorted(list(selected_set - present_inputs))[:10]
        miss_lb = sorted(list(selected_set - present_labels))[:10]
        print(f"[DIAG] Examples missing in inputs (first 10): {miss_in}")
        print(f"[DIAG] Examples missing in labels (first 10): {miss_lb}")

    token_now = extract_now_from_path(args.splits)
    out_path = args.out or f"data/splits/splits_verified_{token_now}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"mode": mode, "selected_tile_keys": have_both}, f, indent=2)

    print(f"[OK] Wrote verified selection → {out_path}")
    print(f"     Kept: {len(have_both)} / {len(selected)}  (dropped: {dropped})")

if __name__ == "__main__":
    main()
