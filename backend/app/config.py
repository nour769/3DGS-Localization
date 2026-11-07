import os
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]


def _configured_path(value: str, *, base: Path = BACKEND_DIR) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def project_data_paths(root: str):
    base = _configured_path(root) if root.strip() else None
    return {
        "project_data_root": str(base) if base else "",
        "colmap_model_path": str(base / "kitprojet" / "3DGS" / "sparse" / "0")
        if base
        else "",
        "reference_images_dir": str(base / "kitprojet" / "3DGS" / "data")
        if base
        else "",
        "query_image_path": str(base / "kitprojet" / "data" / "query.jpg")
        if base
        else "",
        "query_calib_path": str(base / "kitprojet" / "data" / "calib.txt")
        if base
        else "",
        "gaussian_ply_path": str(base / "scene-routiere-finale (2).ply")
        if base
        else "",
    }


_DATA_ROOT = os.getenv("DATA_ROOT", "").strip() or os.getenv(
    "PROJECT_DATA_ROOT", ""
).strip()
if not _DATA_ROOT:
    _DATA_ROOT = str(BACKEND_DIR / ".." / ".." / "data")
_PROJECT_DATA_PATHS = project_data_paths(_DATA_ROOT)


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "local-development-only")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(Path(__file__).resolve().parents[1] / 'instance' / 'localizations.db').as_posix()}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
    PORT = int(os.getenv("PORT", "5000"))
    DEBUG = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    UPLOAD_DIR = _configured_path(os.getenv("UPLOAD_DIR", "./instance/uploads"))
    SCENES_DIR = os.getenv("SCENES_DIR", "./data/scenes")
    DATA_ROOT = _DATA_ROOT
    PROJECT_DATA_ROOT = _PROJECT_DATA_PATHS["project_data_root"]
    COLMAP_MODEL_PATH = os.getenv(
        "COLMAP_MODEL_PATH",
        "",
    ) or _PROJECT_DATA_PATHS["colmap_model_path"]
    GAUSSIAN_PLY_PATH = os.getenv(
        "GAUSSIAN_PLY_PATH", ""
    ) or _PROJECT_DATA_PATHS["gaussian_ply_path"]
    REFERENCE_IMAGES_DIR = os.getenv(
        "REFERENCE_IMAGES_DIR",
        "",
    ) or _PROJECT_DATA_PATHS["reference_images_dir"]
    QUERY_IMAGE_PATH = os.getenv(
        "QUERY_IMAGE_PATH",
        "",
    ) or _PROJECT_DATA_PATHS["query_image_path"]
    QUERY_CALIB_PATH = os.getenv(
        "QUERY_CALIB_PATH",
        "",
    ) or _PROJECT_DATA_PATHS["query_calib_path"]
    QUERY_FX = float(os.getenv("QUERY_FX", "716.614"))
    QUERY_FY = float(os.getenv("QUERY_FY", "716.614"))
    QUERY_CX = float(os.getenv("QUERY_CX", "621"))
    QUERY_CY = float(os.getenv("QUERY_CY", "187.5"))
    QUERY_WIDTH = int(os.getenv("QUERY_WIDTH", "1242"))
    QUERY_HEIGHT = int(os.getenv("QUERY_HEIGHT", "375"))
    IMAGE_WORK_WIDTH = int(os.getenv("IMAGE_WORK_WIDTH", "960"))
    MIN_PNP_INLIERS = int(os.getenv("MIN_PNP_INLIERS", "15"))
    PNP_REPROJECTION_ERROR = float(os.getenv("PNP_REPROJECTION_ERROR", "4.0"))
    ENABLE_DINOV2 = os.getenv("ENABLE_DINOV2", "0").lower() in {"1", "true", "yes"}
    DINO_MODEL = os.getenv("DINO_MODEL", "dinov2_vits14")
    SCENE_FRAME_VERIFIED = os.getenv(
        "SCENE_FRAME_VERIFIED", "0"
    ).lower() in {"1", "true", "yes"}
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
