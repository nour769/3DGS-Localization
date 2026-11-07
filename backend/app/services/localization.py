import secrets
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ..extensions import db
from ..localization.engine import LocalizationError, LocalizationUnavailable
from ..models import Localization

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def validate_image(data: bytes):
    if not data:
        raise ValueError("The uploaded image is empty.")
    try:
        with Image.open(data_stream(data)) as image:
            image.verify()
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Upload a valid JPEG, PNG, WebP, or BMP image.") from exc
    if width < 32 or height < 32:
        raise ValueError("Image dimensions must be at least 32 × 32 pixels.")
    return width, height


def data_stream(data):
    from io import BytesIO

    return BytesIO(data)


def save_upload(data: bytes, filename: str, upload_dir: Path):
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported image extension.")
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{secrets.token_hex(16)}{suffix}"
    path.write_bytes(data)
    return path


def create_record(filename: str, image_path: Path, scene_id: str):
    record = Localization(
        image_name=Path(filename).name[:255] or "uploaded-image",
        image_path=str(image_path),
        scene_id=scene_id,
        status="processing",
    )
    db.session.add(record)
    db.session.commit()
    return record


def execute_localization(record, scene, engine):
    try:
        result = engine.localize(record.image_path, scene)
        record.status = result["status"]
        record.pose_json = result.get("pose")
        record.metrics_json = result.get("metrics") or {}
        record.error_message = result.get("message")
        db.session.commit()
        return result
    except LocalizationUnavailable as exc:
        record.status = "failed"
        record.error_message = str(exc)
        record.metrics_json = {}
        db.session.commit()
        raise
    except LocalizationError as exc:
        record.status = "failed"
        record.error_message = str(exc)
        record.metrics_json = {}
        db.session.commit()
        raise
