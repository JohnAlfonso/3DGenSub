#!/usr/bin/env python3
"""Upload a .glb file to the render service and save returned PNG.

Usage:
  python render_glb.py input.glb --out out.png --url http://localhost:9000/render_glb --device cuda

The script posts the file as multipart/form-data to the `/render_glb` endpoint
and writes the returned image bytes to the specified output file.
"""
from __future__ import annotations
import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload .glb file or folder to render service and save PNG(s)")
    p.add_argument("--input", help="Path to input .glb file or folder containing .glb files")
    p.add_argument("--out", "-o", dest="output", help="Output file or directory (default: same folder as input) ")
    p.add_argument("--url", default="http://localhost:9000/render_glb", help="Render service URL")
    p.add_argument("--device", choices=("cuda", "cpu"), help="Device to request from service")
    p.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds")
    return p.parse_args()


def find_glb_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files = [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() == ".glb"]
    return files


def upload_and_save(session, url: str, path: Path, out_path: Path, device: str | None, timeout: float) -> bool:
    with path.open("rb") as fh:
        files = {"file": (path.name, fh, "model/gltf-binary")}
        data = {}
        if device:
            data["device"] = device
        try:
            resp = session.post(url, files=files, data=data or None, timeout=timeout)
        except Exception as e:
            print(f"Request failed for {path}: {e}", file=sys.stderr)
            return False

    if resp.status_code != 200:
        print(f"Server returned {resp.status_code} for {path}: {resp.text}", file=sys.stderr)
        return False

    ctype = resp.headers.get("content-type", "")
    if not ctype.startswith("image/"):
        print(f"Unexpected content-type for {path}: {ctype}", file=sys.stderr)
        return False

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as out_f:
            out_f.write(resp.content)
    except OSError as e:
        print(f"Failed to write {out_path}: {e}", file=sys.stderr)
        return False

    print(f"Saved render: {out_path}")
    return True


def main() -> int:
    args = parse_args()
    inp = Path(args.input)
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 2

    try:
        import requests
    except Exception:
        print("Missing dependency: requests. Install with: pip install requests", file=sys.stderr)
        return 3

    # Determine list of .glb files
    glb_files = find_glb_files(inp)
    if not glb_files:
        print(f"No .glb files found in {inp}", file=sys.stderr)
        return 2

    # Determine output paths
    out_arg = args.output
    if out_arg:
        out_path = Path(out_arg)
        if inp.is_file():
            # single input file -> output can be file or directory
            if out_path.is_dir():
                out_dir = out_path
            else:
                out_dir = out_path.parent
        else:
            # input is folder -> output should be directory
            out_dir = out_path
    else:
        out_dir = inp.parent if inp.is_file() else inp

    successes = 0
    failures = 0
    session = requests.Session()
    for p in glb_files:
        target_name = p.stem + ".png"
        target = out_dir / target_name
        ok = upload_and_save(session, args.url, p, target, args.device, args.timeout)
        if ok:
            successes += 1
        else:
            failures += 1

    print(f"Done. success={successes} fail={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
