import os
import argparse

def run_command(cmd):
    print(f"\n[RUNNING] {cmd}")
    status = os.system(cmd)
    if status != 0:
        raise RuntimeError(f"[ERROR] Command failed: {cmd}")

def main(args):
    os.makedirs("data/pseudo", exist_ok=True)
    os.makedirs("outputs/sam_masks", exist_ok=True)

    run_command("python scripts/select_pseudo_tiles.py")
    run_command("python scripts/run_sam_on_tiles.py")
    run_command(f"python scripts/merge_pseudo_into_train_split.py --split {args.split_path}")

    print(f"\n✅ Done. Updated split saved with pseudo-labeled tiles.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", type=str, required=True,
                        help="Path to existing split JSON (e.g. data/splits/splits_20250721_105734.json)")
    args = parser.parse_args()
    main(args)
