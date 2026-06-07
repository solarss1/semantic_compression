"""Локальне встановлення U²-Net (код + ваги, без torch.hub)."""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from src.utils.file_lock import exclusive_file_lock

ROOT = Path(__file__).resolve().parents[2]
U2NET_REPO = ROOT / "checkpoints" / "U-2-Net"
U2NET_ZIP = ROOT / "checkpoints" / "U-2-Net-master.zip"
LOCK_FILE = ROOT / "checkpoints" / ".u2net_setup.lock"
WEIGHTS_PATH = U2NET_REPO / "saved_models" / "u2net" / "u2net.pth"
ZIP_URL = "https://github.com/xuebinqin/U-2-Net/archive/refs/heads/master.zip"
WEIGHTS_URL = (
    "https://huggingface.co/lilpotat/pytorch3d/resolve/main/u2net.pth"
)
MIN_WEIGHTS_BYTES = 150_000_000


def u2net_repo_ready() -> bool:
    return (U2NET_REPO / "model" / "u2net.py").exists()


def weights_ready() -> bool:
    return WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size >= MIN_WEIGHTS_BYTES


def _extract_zip(zip_path: Path, dest_parent: Path) -> Path:
    dest_parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_parent)
    extracted = dest_parent / "U-2-Net-master"
    if not (extracted / "model" / "u2net.py").exists():
        raise RuntimeError(f"model/u2net.py not found after extract: {extracted}")
    return extracted


def _adopt_extracted_repo(extracted: Path) -> None:
    if extracted.resolve() == U2NET_REPO.resolve():
        return
    if U2NET_REPO.exists():
        shutil.rmtree(U2NET_REPO)
    shutil.move(str(extracted), str(U2NET_REPO))


def ensure_u2net_repo(verbose: bool = True) -> Path:
    """Завантажити код U²-Net у checkpoints/U-2-Net (один раз, з lock)."""
    if u2net_repo_ready():
        return U2NET_REPO

    leftover = U2NET_REPO.parent / "U-2-Net-master"
    if leftover.exists() and (leftover / "model" / "u2net.py").exists():
        _adopt_extracted_repo(leftover)
        if u2net_repo_ready():
            return U2NET_REPO

    U2NET_REPO.parent.mkdir(parents=True, exist_ok=True)

    with exclusive_file_lock(LOCK_FILE):
        if u2net_repo_ready():
            return U2NET_REPO

        if verbose:
            print("Завантаження коду U²-Net (~60 MB zip)...")

        if not U2NET_ZIP.exists() or U2NET_ZIP.stat().st_size < 10_000:
            urlretrieve(ZIP_URL, U2NET_ZIP)  # noqa: S310

        extracted = _extract_zip(U2NET_ZIP, U2NET_REPO.parent)
        _adopt_extracted_repo(extracted)

        if not u2net_repo_ready():
            raise RuntimeError("U²-Net repo setup failed")

        if verbose:
            print(f"  Код U²-Net: {U2NET_REPO}")
        return U2NET_REPO


def ensure_u2net_weights(verbose: bool = True) -> Path:
    """Завантажити u2net.pth (~176 MB)."""
    ensure_u2net_repo(verbose=False)
    if weights_ready():
        return WEIGHTS_PATH

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with exclusive_file_lock(LOCK_FILE):
        if weights_ready():
            return WEIGHTS_PATH

        if verbose:
            print("Завантаження ваг u2net.pth (~176 MB)...")

        tmp = WEIGHTS_PATH.with_suffix(".pth.download")
        urlretrieve(WEIGHTS_URL, tmp)  # noqa: S310
        if tmp.stat().st_size < MIN_WEIGHTS_BYTES:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"u2net.pth занадто малий ({tmp.stat().st_size} bytes)"
            )
        tmp.replace(WEIGHTS_PATH)

        if verbose:
            print(f"  Ваги: {WEIGHTS_PATH}")
        return WEIGHTS_PATH


def load_u2net_model(device: str, *, verbose: bool = False):
    """Завантажити U2NET з локального репозиторію."""
    import torch

    repo = ensure_u2net_repo(verbose=verbose)
    weights = ensure_u2net_weights(verbose=verbose)

    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from model.u2net import U2NET  # noqa: WPS433

    net = U2NET(3, 1)
    try:
        state = torch.load(weights, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights, map_location=device)
    net.load_state_dict(state)
    return net.to(device).eval()


def prepare_roi_models(
    config: dict | None = None,
    *,
    method: str | None = None,
    verbose: bool = False,
) -> None:
    if not needs_u2net(config, method):
        return
    ensure_u2net_repo(verbose=verbose)
    ensure_u2net_weights(verbose=verbose)


def needs_u2net(config: dict | None = None, method: str | None = None) -> bool:
    if method:
        m = method.lower()
        if m in ("ultralytics_sam", "sam_ultralytics", "sam"):
            return False
        if "u2net" in m or m == "combined":
            return True
        if m == "saliency_resnet":
            return False
        if m == "saliency":
            if config:
                return (
                    str(config.get("roi", {}).get("saliency_model", "u2net")).lower()
                    == "u2net"
                )
            return True
        return False
    if not config:
        return False
    roi = config.get("roi", {})
    m = str(roi.get("method", "")).lower()
    if m in ("ultralytics_sam", "sam_ultralytics", "sam"):
        return False
    if "u2net" in m or m == "combined":
        return True
    if m in ("saliency",) or m == "":
        return str(roi.get("saliency_model", "u2net")).lower() == "u2net"
    return False


if __name__ == "__main__":
    prepare_roi_models(method="saliency_u2net", verbose=True)
    load_u2net_model("cpu", verbose=False)
    print("U²-Net готовий.")
