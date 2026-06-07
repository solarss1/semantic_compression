#!/usr/bin/env python3
"""Завантажити нестиснені PNG-зразки для GUI (Kodak + benchmark)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KODAK_MIRRORS = [
    "https://r0k.us/graphics/kodak/kodak",
    "http://r0k.us/graphics/kodak/kodak",
]
KODAK_COUNT = 24
SAMPLES_DIR = ROOT / "data" / "samples"


def _download(url: str, dest: Path, timeout: int = 60) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    print(f"  {dest.name} …")
    try:
        with urlopen(url, timeout=timeout) as response:
            data = response.read()
        if len(data) < 1024:
            return False
        dest.write_bytes(data)
        return True
    except URLError as exc:
        print(f"    помилка: {exc}")
        return False


def download_kodak(dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(1, KODAK_COUNT + 1):
        name = f"kodim{i:02d}.png"
        path = dest / name
        if path.exists() and path.stat().st_size > 1024:
            paths.append(path)
            continue
        ok = False
        for base in KODAK_MIRRORS:
            if _download(f"{base}/{name}", path):
                ok = True
                break
        if ok:
            paths.append(path)
    print(f"  Kodak: {len(paths)}/{KODAK_COUNT}")
    return paths


def export_skimage_benchmark(dest: Path) -> list[Path]:
    """Кольорові PNG 768×512 з scikit-image (офлайн, без мережі)."""
    from skimage import data
    from skimage.color import gray2rgb
    from skimage.transform import resize
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    items = [
        (data.astronaut, "bench_skimage_astronaut.png"),
        (data.coffee, "bench_skimage_coffee.png"),
        (data.chelsea, "bench_skimage_chelsea.png"),
        (data.cat, "bench_skimage_cat.png"),
        (data.hubble_deep_field, "bench_skimage_hubble.png"),
        (data.grass, "bench_skimage_grass.png"),
        (data.brick, "bench_skimage_brick.png"),
    ]
    target = (512, 768)
    paths: list[Path] = []

    for loader, filename in items:
        path = dest / filename
        if path.exists():
            paths.append(path)
            continue
        try:
            arr = loader()
        except (AttributeError, FileNotFoundError, OSError) as exc:
            print(f"  пропуск {filename}: {exc}")
            continue
        arr = arr.astype("uint8")
        if arr.ndim == 2:
            arr = gray2rgb(arr)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        h, w = arr.shape[:2]
        th, tw = target
        scale = max(th / h, tw / w)
        new_h, new_w = max(th, int(round(h * scale))), max(tw, int(round(w * scale)))
        scaled = resize(arr, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(
            "uint8"
        )
        y0, x0 = (new_h - th) // 2, (new_w - tw) // 2
        cropped = scaled[y0 : y0 + th, x0 : x0 + tw]
        Image.fromarray(cropped).save(path, optimize=True)
        paths.append(path)

    print(f"  Benchmark: {len(paths)}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Завантажити PNG-зразки для GUI")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=SAMPLES_DIR,
        help="Каталог зразків (за замовчуванням data/samples)",
    )
    parser.add_argument("--kodak-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    do_kodak = not args.benchmark_only
    do_bench = not args.kodak_only

    total = 0
    if do_kodak:
        print("Kodak (PNG, мережа)…")
        total += len(download_kodak(out))
    if do_bench:
        print("Benchmark scikit-image (PNG, офлайн)…")
        total += len(export_skimage_benchmark(out))

    print(f"\nГотово: {out} ({total} нових/існуючих файлів)")


if __name__ == "__main__":
    main()
