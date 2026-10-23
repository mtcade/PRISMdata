#!/usr/bin/env python3
"""Download and extract data.zip from Zenodo into data/.

Usage:
    uv run python scripts/fetch_data.py

Not run automatically by `uv sync` or by importing `prismdata` -- data
acquisition is a separate, explicit step from installing the package.
Safe to re-run: if data.zip already exists locally with the expected
checksums (SHA-256 and MD5), it's reused instead of re-downloaded.
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
SHA256 = "1865a0f3bb6fb917298c5a26b13939ed4eeb7eb021b6d6c0ecccadb538000de3"
MD5 = "a4dce505b02f60d876d057486de4e9d3"


def _digest_of(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of(path: Path) -> str:
    return _digest_of(path, "sha256")


def md5_of(path: Path) -> str:
    return _digest_of(path, "md5")


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
    parser.add_argument("--md5", default=MD5, help="expected MD5 checksum of data.zip")
    parser.add_argument(
        "--dest", type=Path, default=REPO_ROOT, help="directory to extract data/ into"
    )
    parser.add_argument(
        "--keep-zip", action="store_true", help="don't delete data.zip after extraction"
    )
    args = parser.parse_args()

    zip_path = args.dest / "data.zip"

    if zip_path.exists():
        print(f"{zip_path} already exists, checking checksums...")
        if sha256_of(zip_path) == args.sha256 and md5_of(zip_path) == args.md5:
            print("checksums match, skipping download")
        else:
            print("checksum mismatch, re-downloading")
            download(args.url, zip_path)
    else:
        download(args.url, zip_path)

    actual_sha256 = sha256_of(zip_path)
    actual_md5 = md5_of(zip_path)
    if actual_sha256 != args.sha256 or actual_md5 != args.md5:
        print(
            f"error: checksum mismatch\n"
            f"  expected sha256 {args.sha256}\n"
            f"  got      sha256 {actual_sha256}\n"
            f"  expected md5    {args.md5}\n"
            f"  got      md5    {actual_md5}",
            file=sys.stderr,
        )
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
