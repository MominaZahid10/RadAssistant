"""
Tests for image storage layout (Phase 4 Step 1).

The important ones are the path-traversal tests. Storage paths come from the
database, and a bug that let one escape the storage root would turn an image
endpoint into arbitrary file read — the single worst defect available in this
phase.
"""

import uuid
from datetime import datetime, timezone

import pytest

from app.services import image_storage


@pytest.fixture(autouse=True)
def temp_storage(tmp_path, monkeypatch):
    """Point IMAGE_DIR at a temp directory for every test."""
    monkeypatch.setattr(image_storage.settings, "IMAGE_DIR", str(tmp_path), raising=False)
    return tmp_path


# ══════════════════════════════════════════════════════════════
# PATH TRAVERSAL — the security boundary
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "evil",
    [
        "../../../etc/passwd",
        "../../secrets.env",
        "2026/08/11/../../../../etc/shadow",
        "/etc/passwd",
    ],
)
def test_resolve_rejects_paths_outside_the_root(evil):
    """
    `root / "../../etc/passwd"` resolves happily outside the root. Every read
    and delete goes through resolve(), so this rejection happens once rather
    than in a dozen call sites that might each forget.
    """
    with pytest.raises(ValueError, match="outside the image store"):
        image_storage.resolve(evil)


def test_dot_quad_sequences_are_literal_directories_not_traversal(temp_storage):
    """
    `....//` is the classic bypass for NAIVE sanitisers: code that does
    `path.replace("..", "")` turns `....//` back into `..//`, re-creating the
    traversal it just removed.

    We don't sanitise by string surgery — we resolve the path and check
    containment — so `....` is simply a four-character directory name and the
    result stays inside the root. This test pins that behaviour so nobody
    "improves" resolve() into a string-replacement version later.
    """
    resolved = image_storage.resolve("....//....//etc/passwd")
    assert resolved.is_relative_to(temp_storage.resolve())
    assert "etc/passwd" in str(resolved)      # inside the store, not the system one


def test_resolve_accepts_normal_relative_paths(temp_storage):
    resolved = image_storage.resolve("2026/08/11/abc.png")
    assert resolved.is_relative_to(temp_storage.resolve())


def test_delete_refuses_traversal_instead_of_raising():
    """
    Deletion is usually cleanup after a DB operation. A traversal attempt must
    fail closed (return False), not raise and abort the surrounding cleanup.
    """
    assert image_storage.delete("../../../etc/passwd") is False


def test_exists_refuses_traversal():
    assert image_storage.exists("../../../etc/passwd") is False


# ══════════════════════════════════════════════════════════════
# PATH CONSTRUCTION
# ══════════════════════════════════════════════════════════════


def test_new_path_is_date_sharded():
    """
    A flat directory of 100,000 files is slow to list and painful to inspect.
    Sharding by date keeps directories small and matches how retention
    policies are expressed.
    """
    now = datetime.now(timezone.utc)
    path = image_storage.new_relative_path(uuid.uuid4())
    assert path.startswith(f"{now.year:04d}/{now.month:02d}/{now.day:02d}/")


def test_new_path_uses_the_uuid_not_the_original_filename():
    """
    Two clinicians uploading "scan.dcm" must not collide — and an original
    filename can itself carry PHI ("smith_john_chest.dcm").
    """
    image_id = uuid.uuid4()
    assert str(image_id) in image_storage.new_relative_path(image_id)


def test_new_path_is_relative():
    """
    An absolute path in the database breaks the moment storage moves, and it
    will move: bind mount → volume → object storage.
    """
    path = image_storage.new_relative_path(uuid.uuid4())
    assert not path.startswith("/")
    assert ":" not in path            # no Windows drive letters either


def test_new_path_creates_its_directory(temp_storage):
    path = image_storage.new_relative_path(uuid.uuid4())
    assert (temp_storage / path).parent.is_dir()


def test_suffix_is_normalised():
    image_id = uuid.uuid4()
    assert image_storage.new_relative_path(image_id, "PNG").endswith(".png")
    assert image_storage.new_relative_path(image_id, ".JPG").endswith(".jpg")


def test_paths_use_forward_slashes_on_every_platform():
    """
    These are stored in the database and served over HTTP. Windows backslashes
    would break both.
    """
    assert "\\" not in image_storage.new_relative_path(uuid.uuid4())


def test_thumbnail_path_derives_from_the_image_path():
    thumb = image_storage.thumbnail_path_for("2026/08/11/abc123.png")
    assert thumb == "2026/08/11/abc123.thumb.jpg"


def test_thumbnail_is_always_jpg():
    """JPEG for thumbnails — smaller than PNG for photographic content."""
    for suffix in (".png", ".jpg", ".webp"):
        thumb = image_storage.thumbnail_path_for(f"2026/08/11/x{suffix}")
        assert thumb.endswith(".thumb.jpg")


# ══════════════════════════════════════════════════════════════
# READ / WRITE / DELETE
# ══════════════════════════════════════════════════════════════


def test_write_then_read_roundtrip():
    path = image_storage.new_relative_path(uuid.uuid4())
    data = b"\x89PNG\r\n\x1a\n" + b"payload"

    assert image_storage.write_bytes(path, data) == len(data)
    assert image_storage.read_bytes(path) == data
    assert image_storage.exists(path) is True


def test_delete_removes_the_file():
    path = image_storage.new_relative_path(uuid.uuid4())
    image_storage.write_bytes(path, b"data")

    assert image_storage.delete(path) is True
    assert image_storage.exists(path) is False


def test_deleting_a_missing_file_is_not_an_error():
    """
    Deletion is part of cleaning up a database row. Failing because the file
    was already gone would block that cleanup and orphan the row instead.
    """
    assert image_storage.delete("2026/08/11/never-existed.png") is False


def test_storage_stats_counts_files():
    for _ in range(3):
        image_storage.write_bytes(image_storage.new_relative_path(uuid.uuid4()), b"xxxx")

    stats = image_storage.storage_stats()
    assert stats["files"] == 3
    assert stats["bytes"] == 12


def test_storage_root_is_created_on_demand(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "images"
    monkeypatch.setattr(image_storage.settings, "IMAGE_DIR", str(target), raising=False)

    assert image_storage.storage_root().is_dir()


# ══════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════


def test_medical_image_model_defaults():
    from app.models.image import MedicalImage

    img = MedicalImage(filename="chest.dcm", storage_path="2026/08/11/x.png")
    cols = MedicalImage.__table__.columns

    # ⚠️  is_deidentified MUST default to False. Defaulting to True would mean
    # a pipeline bug silently produces images labelled safe that still carry
    # PHI — the worst failure available in this phase.
    assert cols["is_deidentified"].default.arg is False
    assert cols["status"].default.arg == "processing"
    assert cols["source_type"].default.arg == "image_upload"


def test_document_id_is_nullable():
    """
    A DICOM study isn't a text document. Nullable by design, not omission.
    """
    from app.models.image import MedicalImage
    assert MedicalImage.__table__.columns["document_id"].nullable is True


def test_every_fk_sets_null_rather_than_cascading():
    """
    Deleting an article must not silently destroy its figures, and deleting a
    user must not destroy their uploads. An orphaned image is recoverable; a
    deleted one is not.

    ⚠️  ASSERTS THE BEHAVIOUR, NOT THE COUNT.
    This used to read `len(fks) == 1`, which broke the moment Phase 6 added
    user ownership — failing for a reason that had nothing to do with cascade
    behaviour. Counting foreign keys tests the schema's shape; what actually
    matters is that none of them cascades. Checking every FK also means a
    future one is covered without anyone remembering to update a number.
    """
    from app.models.image import MedicalImage

    fks = list(MedicalImage.__table__.foreign_keys)
    assert fks, "the image table should reference something"

    for fk in fks:
        assert fk.ondelete == "SET NULL", (
            f"{fk.target_fullname} cascades — deleting it would destroy images"
        )

    # The two relationships that exist today, named so a removal is visible.
    targets = {fk.column.table.name for fk in fks}
    assert {"documents", "users"} <= targets


def test_study_date_is_a_date_not_a_datetime():
    """
    DICOM StudyDate has no time component. Storing a fake midnight implies
    precision that isn't there.
    """
    import sqlalchemy as sa
    from app.models.image import MedicalImage

    col = MedicalImage.__table__.columns["study_date"]
    assert isinstance(col.type, sa.Date)
    assert not isinstance(col.type, sa.DateTime)
