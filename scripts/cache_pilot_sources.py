#!/usr/bin/env python3
"""Download immutable pilot source files into the local preprocessing cache."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from state_model_interface.prepare_pilot import DEFAULT_SOURCES, _cached_url_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def download(url: str, cache: Path) -> dict[str, object]:
    output = _cached_url_path(url, cache)
    partial = output.with_suffix(output.suffix + ".partial")
    if not output.is_file():
        subprocess.run(
            [
                "curl",
                "--location",
                "--fail",
                "--retry",
                "8",
                "--retry-all-errors",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            check=True,
        )
        os.replace(partial, output)
    result = {"url": url, "path": str(output), "size": output.stat().st_size}
    print(f"cached {result['size']} bytes: {url}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    args.cache.mkdir(parents=True, exist_ok=True)
    urls = [
        url
        for source in DEFAULT_SOURCES
        if source.data_url
        for url in (
            (source.data_url,) if isinstance(source.data_url, str) else source.data_url
        )
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(lambda url: download(url, args.cache), urls))
    manifest = {"complete": True, "files": files}
    (args.cache / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"cached {len(files)} immutable source files in {args.cache}")


if __name__ == "__main__":
    main()
