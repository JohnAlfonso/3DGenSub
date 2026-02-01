import argparse
import json
import os
import sys
import requests
from pathlib import Path

BASE_DIR = Path("/root/subnet/404/404-competition-0/rounds")
RESULT_DIR = Path("result")


def parse_args():
    parser = argparse.ArgumentParser(description="Download PNGs")

    parser.add_argument("--round", required=True, type=int, help="Round number")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--coldkey", help="Coldkey directory name")
    group.add_argument("--prompts", action="store_true", help="Download from prompts.txt")

    return parser.parse_args()


def download_png(url: str, output_path: Path):
    if output_path.exists():
        print(f"⏭️  Skipping: {output_path.name}")
        return

    print(f"⬇️  Downloading: {output_path.name}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(r.content)


def download_from_generations(round_dir: Path, coldkey: str, output_dir: Path):
    coldkey_dir = round_dir / coldkey
    json_path = coldkey_dir / "generations.json"

    if not json_path.exists():
        print(f"❌ generations.json not found: {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        generations = json.load(f)

    for _, data in generations.items():
        png_url = data.get("png")
        if not png_url:
            continue

        png_name = os.path.basename(png_url)
        download_png(png_url, output_dir / png_name)


def download_from_prompts(round_dir: Path, output_dir: Path):
    prompts_path = round_dir / "prompts.txt"

    if not prompts_path.exists():
        print(f"❌ prompts.txt not found: {prompts_path}")
        sys.exit(1)

    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url:
                continue

            png_name = os.path.basename(url)
            download_png(url, output_dir / png_name)


def main():
    args = parse_args()

    round_dir = BASE_DIR / str(args.round)
    if not round_dir.exists():
        print(f"❌ Round directory not found: {round_dir}")
        sys.exit(1)

    RESULT_DIR.mkdir(exist_ok=True)

    if args.coldkey:
        output_dir = RESULT_DIR / f"round-{args.round}-{args.coldkey}"
        output_dir.mkdir(parents=True, exist_ok=True)
        download_from_generations(round_dir, args.coldkey, output_dir)

    elif args.prompts:
        output_dir = RESULT_DIR / f"round-{args.round}-prompt"
        output_dir.mkdir(parents=True, exist_ok=True)
        download_from_prompts(round_dir, output_dir)


if __name__ == "__main__":
    main()