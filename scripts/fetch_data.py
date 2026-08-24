#!/usr/bin/env python3
"""Download and extract data.zip from Zenodo into data/.

Usage:
    uv run python scripts/fetch_data.py

Not run automatically by `uv sync` or by importing `prismdata` -- data
acquisition is a separate, explicit step from installing the package.
Safe to re-run: if data.zip already exists locally with the expected
checksum, it's reused instead of re-downloaded.
"""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# DOI 10.5281/zenodo.22085306
ZENODO_URL = "https://zenodo.org/records/22085306/files/data.zip?download=1"
SHA256 = "b97056e94993986bd8c908a25d0bc519164b04e7c29d50374c051a92b6eb0414"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url}")

    def report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        print(f"\r  {done / 1e6:8.1f} / {total_size / 1e6:8.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=ZENODO_URL, help="URL to data.zip on Zenodo")
    parser.add_argument("--sha256", default=SHA256, help="expected SHA-256 checksum of data.zip")
    parser.add_argument(
        "--dest", type=Path, default=REPO_ROOT, help="directory to extract data/ into"
    )
    parser.add_argument(
        "--keep-zip", action="store_true", help="don't delete data.zip after extraction"
    )
    args = parser.parse_args()

    zip_path = args.dest / "data.zip"

    if zip_path.exists():
        print(f"{zip_path} already exists, checking checksum...")
        if sha256_of(zip_path) == args.sha256:
            print("checksum matches, skipping download")
        else:
            print("checksum mismatch, re-downloading")
            download(args.url, zip_path)
    else:
        download(args.url, zip_path)

    actual = sha256_of(zip_path)
    if actual != args.sha256:
        print(f"error: checksum mismatch\n  expected {args.sha256}\n  got      {actual}", file=sys.stderr)
        return 1

    print(f"Extracting to {args.dest}/data/")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.dest)

    if not args.keep_zip:
        zip_path.unlink()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
