"""Step 0: download the raw data and verify checksums.

Sources
-------
- Figshare article 20123153 (Kromp et al. 2023): clinical annotations (+ images, optional).
- GitHub repository of the dataset, pinned commit: expert grades of all 2,344 images
  (``complete.csv``) and the 11 annotators + consensus on the 300-image test set.

The raw files are never edited. If a download is blocked (proxy, offline), place the files
manually in ``data/raw/`` with the names below; checksums are still verified.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import requests

from .config import Paths, get_logger

# MD5 of the files as used for the published analysis (verified 2026-11-21).
EXPECTED_MD5 = {
    "Clincial_annotations.csv": "f136008a6dff2abcd774f6ab4d3fa74f",
    "complete.csv": "b6c4a75fa1b00bf44a0a4231d7290f62",
    "annotations/test_rev.csv": "c510c01a5285bd6daea6e48bf8e72c6a",
    "annotations/Gardner_Expert.csv": "df2a0ff5f74b7ffe859fc7dba2afa334",
    "annotations/Annotator_0.csv": "7eb9ca4414116ea63f0f0c5eeaa3c5c6",
    "annotations/Annotator_1.csv": "1b860c5f117026049c2d153ada745960",
    "annotations/Annotator_2.csv": "0366c4a44cc07a02f9f3bb3f4b954117",
    "annotations/Annotator_3.csv": "9788e0cfca05d2a57ef98197e8013a67",
    "annotations/Annotator_4.csv": "8f717cd6c940b4c1d00e2b550e8a1967",
    "annotations/Annotator_5.csv": "fca2ff13ecc5716fb186d5db6b49d90e",
    "annotations/Annotator_6.csv": "34672d55c016bd1141d2aa4d9a91e788",
    "annotations/Annotator_7.csv": "853a111352b244b8424978ea51eb980a",
    "annotations/Annotator_8.csv": "093b5d16a838f6630d9f40000eea6525",
    "annotations/Annotator_9.csv": "120e7bfb7443e307551c7013b6c95f0e",
    "annotations/Annotator_10.csv": "744e0f2ab859e319972a6adb3587562d",
}
GITHUB_FILES = [k for k in EXPECTED_MD5 if k != "Clincial_annotations.csv"]
FIGSHARE_FILES = ["Clincial_annotations.csv"]  # sic: typo in the original file name


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.replace(dest)


def figshare_files(article_id: int) -> list[dict]:
    r = requests.get(f"https://api.figshare.com/v2/articles/{article_id}", timeout=60)
    r.raise_for_status()
    return r.json()["files"]


def download_all(cfg: dict, paths: Paths, images: bool = False) -> None:
    log = get_logger("00_download", paths)
    raw = paths.raw

    # 1) GitHub files (pinned commit)
    repo, ref = cfg["data"]["github_repo"], cfg["data"]["github_ref"]
    for rel in GITHUB_FILES:
        dest = raw / rel
        if dest.exists():
            continue
        url = f"https://raw.githubusercontent.com/{repo}/{ref}/{rel}"
        log.info("GitHub  -> %s", rel)
        _get(url, dest)

    # 2) Figshare files
    wanted = [f for f in FIGSHARE_FILES if not (raw / f).exists()]
    if wanted or images:
        files = figshare_files(cfg["data"]["figshare_article_id"])
        for f in files:
            name = f["name"]
            is_image_bundle = name.lower().endswith(".zip") and "image" in name.lower()
            if name in wanted or (images and is_image_bundle):
                dest = raw / name
                if not dest.exists():
                    log.info("Figshare -> %s (%.1f MB)", name, f.get("size", 0) / 1e6)
                    _get(f["download_url"], dest, timeout=600)
                if f.get("computed_md5") and md5(dest) != f["computed_md5"]:
                    raise RuntimeError(f"Figshare checksum mismatch for {name}")
                if is_image_bundle:
                    with zipfile.ZipFile(dest) as z:
                        z.extractall(raw)
                    log.info("Extracted %s", name)

    verify(paths)


def verify(paths: Paths) -> bool:
    """Check presence and MD5 of every required raw file. Mismatch = warning (logged)."""
    log = get_logger("00_download", paths)
    ok = True
    lines = ["file,md5,expected,status"]
    for rel, exp in EXPECTED_MD5.items():
        p = paths.raw / rel
        if not p.exists():
            log.error("MISSING %s - download it or place it manually in data/raw/", rel)
            ok = False
            lines.append(f"{rel},,{exp},missing")
            continue
        got = md5(p)
        status = "ok" if got == exp else "MISMATCH"
        if status != "ok":
            log.warning("Checksum differs for %s (got %s, expected %s)", rel, got, exp)
        lines.append(f"{rel},{got},{exp},{status}")
    (paths.raw / "MANIFEST.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Raw data verification: %s", "all files present" if ok else "files missing")
    return ok
