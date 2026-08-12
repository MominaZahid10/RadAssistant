"""
RadAssist AI — Image Storage Layout (Phase 4)

Owns where image files live on disk and how paths are constructed. Everything
else in the codebase deals in *relative* paths and asks this module to resolve
them.

WHY A DEDICATED MODULE FOR SOMETHING THIS SMALL:
Path handling is where directory-traversal bugs live. If callers built paths
themselves, one that interpolated a user-supplied filename would let
`../../etc/passwd` escape the storage root. Resolution happens in exactly one
place, and that place validates.

LAYOUT — sharded by date, then by image id:

    images/
      2026/08/11/
        3f2a1c9e-....png          full image
        3f2a1c9e-....thumb.jpg    256px thumbnail

WHY SHARD BY DATE:
A flat directory with 100,000 files is slow to list on most filesystems and
painful to inspect by hand. Date sharding keeps directories small, makes
"what arrived on the 11th" trivial, and matches how retention policies are
usually expressed.

WHY THE ID IN THE FILENAME AND NOT THE ORIGINAL NAME:
Two clinicians both uploading "scan.dcm" must not collide, and an original
filename may itself contain PHI ("smith_john_chest.dcm"). The database keeps
the original name for display; the disk uses the UUID.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings

settings = get_settings()

# Thumbnails: 256px on the long edge. Big enough to recognise a chest film in
# the evidence panel, small enough that a gallery of 50 loads instantly.
THUMBNAIL_MAX_PX = 256

# Extensions we will write. Anything else is converted to PNG on ingest —
# DICOM pixel data has no native web format, and browsers can't display TIFF.
WEB_SAFE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def storage_root() -> Path:
    """Absolute root for all image files. Created on first use."""
    root = Path(settings.IMAGE_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_relative_path(image_id: uuid.UUID, suffix: str = ".png") -> str:
    """
    Build the relative path for a new image, creating its directory.

    Returns something like "2026/08/11/3f2a1c9e-....png" — relative, because
    an absolute path stored in the database breaks the moment the storage root
    moves, and it will move (bind mount → volume → S3).
    """
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    suffix = suffix.lower()

    now = datetime.now(timezone.utc)
    shard = Path(f"{now.year:04d}") / f"{now.month:02d}" / f"{now.day:02d}"
    (storage_root() / shard).mkdir(parents=True, exist_ok=True)

    return str(shard / f"{image_id}{suffix}").replace("\\", "/")


def thumbnail_path_for(relative_path: str) -> str:
    """Thumbnail path derived from an image's path — always .jpg."""
    p = Path(relative_path)
    return str(p.with_name(f"{p.stem}.thumb.jpg")).replace("\\", "/")


def resolve(relative_path: str) -> Path:
    """
    Turn a stored relative path into an absolute one, safely.

    ⚠️  THE SECURITY CHECK IS THE POINT OF THIS FUNCTION.
    `storage_root() / "../../etc/passwd"` resolves happily outside the root.
    Every read and delete goes through here, so a path that escapes is
    rejected once rather than in a dozen call sites that each might forget.
    """
    root = storage_root().resolve()
    candidate = (root / relative_path).resolve()

    if not candidate.is_relative_to(root):
        raise ValueError(
            f"Refusing to access a path outside the image store: {relative_path!r}"
        )
    return candidate


def write_bytes(relative_path: str, data: bytes) -> int:
    """Write file contents. Returns bytes written."""
    target = resolve(relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return len(data)


def read_bytes(relative_path: str) -> bytes:
    return resolve(relative_path).read_bytes()


def exists(relative_path: str) -> bool:
    try:
        return resolve(relative_path).is_file()
    except ValueError:
        return False


def delete(relative_path: str) -> bool:
    """
    Remove a file. Missing files are not an error.

    Deletion is usually part of cleaning up a database row; failing because
    the file was already gone would block that cleanup and leave the row
    orphaned instead.
    """
    try:
        target = resolve(relative_path)
    except ValueError:
        return False

    if target.is_file():
        target.unlink()
        return True
    return False


def storage_stats() -> dict:
    """Counts and total size — for /health and the stats endpoint."""
    root = storage_root()
    files = [p for p in root.rglob("*") if p.is_file()]
    return {
        "root": str(root),
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }
