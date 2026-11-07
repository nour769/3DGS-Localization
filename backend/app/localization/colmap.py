from pathlib import Path


class ColmapModelError(RuntimeError):
    pass


def load_reconstruction(model_path: Path):
    try:
        import pycolmap
    except ImportError as exc:
        raise ColmapModelError(
            "pycolmap is not installed. Install backend/requirements.txt to load COLMAP."
        ) from exc

    if not model_path.is_dir():
        raise ColmapModelError(f"COLMAP model directory not found: {model_path}")
    try:
        reconstruction = pycolmap.Reconstruction(str(model_path))
    except (RuntimeError, ValueError, OSError) as exc:
        raise ColmapModelError(
            f"Could not read the COLMAP model at {model_path}: {exc}"
        ) from exc
    if not reconstruction.images or not reconstruction.points3D:
        raise ColmapModelError("The COLMAP model has no registered images or 3D points.")
    return reconstruction


def model_image_count(model_path: Path):
    for filename in ("images.bin", "images.txt"):
        if (model_path / filename).is_file():
            return None
    return None
