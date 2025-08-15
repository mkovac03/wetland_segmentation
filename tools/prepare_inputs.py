#!/usr/bin/env python3
# Parallel merger with tqdm:
# - Reads bands_00 (no label), bands_01_22 (label in band 0), bands_23_44, bands_45_63
# - Uses label only from bands_01_22; drops label bands from other folders
# - Skips tiles that are ALL -32768 across all channels
# - Per-pixel: if ALL channels are -32768, label -> 255
# - Scales valid embeddings by /10000 (keeps -32768 sentinel)
# - Saves inputs (64 bands, float32, nodata=-32768) and labels (uint8, nodata=255)
# - Multiprocessing + tqdm; --workers -1 uses all cores, 0 uses (cores-1)

import os, re, argparse, numpy as np, rasterio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- Defaults (override with CLI if needed) ----
DIR_00    = "/media/lkm413/storage21/gee_embedding_download/images/Europe/2018/bands_00"
DIR_01_22 = "/media/lkm413/storage21/gee_embedding_download/images/Europe/2018/bands_01_22"
DIR_23_44 = "/media/lkm413/storage21/gee_embedding_download/images/Europe/2018/bands_23_44"
DIR_45_63 = "/media/lkm413/storage21/gee_embedding_download/images/Europe/2018/bands_45_63"

OUT_X_DIR = "/media/lkm413/storage11/wetland_segmentation/data/inputs"
OUT_Y_DIR = "/media/lkm413/storage11/wetland_segmentation/data/labels"

NODATA_SENTINEL = -32768
SCALE_DIVISOR   = 10000.0
LABEL_NODATA    = 255

LABEL_REMAP = {
    0:0, 1:1, 2:2, 4:2, 6:2, 8:3, 9:4, 10:5, 12:6, 11:7, 13:7,
    14:8, 15:9, 16:10, 17:10, 20:10, 21:10, 22:10, 18:11, 19:12,
}

RX_ANY = re.compile(
    r"google_embed_Europe_2018_10m_(?P<utm>\d+)_bands_(?P<rng>00|01_22|23_44|45_63)_(?P<tile>\d+)\.(?:tif|tiff)$"
)

def normalize_workers(n: int | None) -> int:
    cpu = os.cpu_count() or 4
    if n is None or n == -1:
        return cpu
    if n == 0:
        return max(1, cpu - 1)
    return max(1, n)

def index_keys(folder):
    keys=set()
    if not os.path.isdir(folder): return keys
    for fn in os.listdir(folder):
        m=RX_ANY.match(fn)
        if m: keys.add(f"{m.group('utm')}_{m.group('tile')}")
    return keys

def read_all(p):
    with rasterio.open(p) as src:
        arr=src.read(); prof=src.profile.copy(); transform=src.transform; crs=src.crs
    return arr, prof, transform, crs

def write_tif(path, arr, profile, *, dtype, nodata):
    prof=profile.copy()
    prof.update(driver="GTiff", count=arr.shape[0], dtype=dtype, nodata=nodata,
                compress="lzw", tiled=True,
                blockxsize=min(256, prof.get("width", arr.shape[2])),
                blockysize=min(256, prof.get("height", arr.shape[1])))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr)

def remap_labels(y):
    out=np.full_like(y, LABEL_NODATA, dtype=np.uint8)
    for s,d in LABEL_REMAP.items(): out[y==s]=np.uint8(d)
    return out

def paths_for_key(key, d00, d12, d34, d56):
    utm, tile = key.split("_",1)
    p00 = os.path.join(d00, f"google_embed_Europe_2018_10m_{utm}_bands_00_{tile}.tif")
    p12 = os.path.join(d12, f"google_embed_Europe_2018_10m_{utm}_bands_01_22_{tile}.tif")
    p34 = os.path.join(d34, f"google_embed_Europe_2018_10m_{utm}_bands_23_44_{tile}.tif")
    p56 = os.path.join(d56, f"google_embed_Europe_2018_10m_{utm}_bands_45_63_{tile}.tif")
    # allow .tiff fallback
    for p_dir, p in ((d00,p00),(d12,p12),(d34,p34),(d56,p56)):
        if not os.path.exists(p):
            alt = p[:-4] + "tiff"
            if os.path.exists(alt):
                if p_dir == d00: p00 = alt
                elif p_dir == d12: p12 = alt
                elif p_dir == d34: p34 = alt
                else: p56 = alt
    return p00, p12, p34, p56, utm, tile

def out_paths(utm, tile, out_x, out_y):
    x = os.path.join(out_x, f"google_embed_Europe_2018_10m_{utm}_bands_00_63_{tile}.tif")
    y = os.path.join(out_y, f"ext_wetland_2018_Europe_2018_10m_{utm}_{tile}.tif")
    return x, y

def process_one(key, d00, d12, d34, d56, out_x, out_y, overwrite, check_label):
    p00,p12,p34,p56,utm,tile = paths_for_key(key, d00,d12,d34,d56)
    if not (os.path.exists(p00) and os.path.exists(p12) and os.path.exists(p34) and os.path.exists(p56)):
        return ("missing", key)

    outx, outy = out_paths(utm, tile, out_x, out_y)
    if (not overwrite) and os.path.exists(outx) and os.path.exists(outy):
        return ("skip", key)

    x00, prof00, transform, crs = read_all(p00)     # (1,H,W)   (no label band)
    x12, prof12, _, _            = read_all(p12)    # (1+22,H,W) label first
    x34, prof34, _, _            = read_all(p34)    # (1+22,H,W) label first (duplicate)
    x56, prof56, _, _            = read_all(p56)    # (1+19,H,W) label first (duplicate)

    H, W = x12.shape[1], x12.shape[2]
    if any(b.shape[1:]!=(H,W) for b in (x00,x34,x56)):
        return ("shape_mismatch", key)

    # take label only once (from 01_22)
    lbl_raw = x12[0].astype(np.int64)
    if check_label and (not np.array_equal(lbl_raw, x34[0]) or not np.array_equal(lbl_raw, x56[0])):
        # just flag; proceed with 01_22
        pass

    # drop label bands from the others
    x12 = x12[1:]  # 22
    x34 = x34[1:]  # 22
    x56 = x56[1:]  # 19
    if not (x00.shape[0]==1 and x12.shape[0]==22 and x34.shape[0]==22 and x56.shape[0]==19):
        return ("band_count_error", key)

    # merge: 1 + 22 + 22 + 19 = 64
    x = np.concatenate([x00, x12, x34, x56], axis=0).astype(np.float32)

    # tile-level all-nodata check
    px_all_nodata = np.all(x == NODATA_SENTINEL, axis=0)
    if np.all(px_all_nodata):
        return ("all_nodata", key)

    # scale valid values, keep sentinel
    mask_valid = (x != NODATA_SENTINEL)
    x[mask_valid] /= SCALE_DIVISOR
    x[~mask_valid] = float(NODATA_SENTINEL)

    # remap labels; mask pixels that are nodata across all bands
    y = remap_labels(lbl_raw).astype(np.uint8)
    y[px_all_nodata] = LABEL_NODATA

    # write
    prof_x = prof12.copy()
    prof_x.update(count=64, dtype="float32", nodata=float(NODATA_SENTINEL),
                  transform=transform, crs=crs)
    write_tif(outx, x, prof_x, dtype="float32", nodata=float(NODATA_SENTINEL))

    prof_y = prof12.copy()
    prof_y.update(count=1, dtype="uint8", nodata=LABEL_NODATA,
                  transform=transform, crs=crs)
    write_tif(outy, y[np.newaxis, ...], prof_y, dtype="uint8", nodata=LABEL_NODATA)

    return ("ok", key)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir00", default=DIR_00)
    ap.add_argument("--dir01_22", default=DIR_01_22)
    ap.add_argument("--dir23_44", default=DIR_23_44)
    ap.add_argument("--dir45_63", default=DIR_45_63)
    ap.add_argument("--out_x", default=OUT_X_DIR)
    ap.add_argument("--out_y", default=OUT_Y_DIR)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_label_check", action="store_true")
    ap.add_argument("--workers", type=int, default=-1, help="-1: all cores, 0: cores-1")
    args = ap.parse_args()

    args.workers = normalize_workers(args.workers)

    os.makedirs(args.out_x, exist_ok=True)
    os.makedirs(args.out_y, exist_ok=True)

    k00 = index_keys(args.dir00)
    k12 = index_keys(args.dir01_22)
    k34 = index_keys(args.dir23_44)
    k56 = index_keys(args.dir45_63)
    keys = sorted(k00 & k12 & k34 & k56)
    if not keys:
        print("[WARN] No matching tiles across all four folders.")
        return

    print(f"[INFO] Found {len(keys)} matching tiles | workers={args.workers}")
    stats = {"ok":0,"skip":0,"missing":0,"shape_mismatch":0,"band_count_error":0,"all_nodata":0,"error":0}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                process_one, key,
                args.dir00, args.dir01_22, args.dir23_44, args.dir45_63,
                args.out_x, args.out_y, args.overwrite, not args.no_label_check
            ): key for key in keys
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Merging", unit="tile"):
            try:
                status, _ = fut.result()
                stats[status] = stats.get(status, 0) + 1
            except Exception:
                stats["error"] += 1

    print("[INFO] Done.",
          " | ".join(f"{k}:{v}" for k,v in stats.items()))

if __name__ == "__main__":
    main()
