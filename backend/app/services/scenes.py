import os
import re
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COLMAP_CORE_FILES = ("cameras", "images", "points3D")
COLMAP_OPTIONAL_FILES = ("rigs", "frames")


def _scene_id(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", path.name).strip("-").lower() or "scene"


class SceneManager:
    def __init__(self, config):
        self.config = config

    def _roots(self):
        roots = [
            Path(item).expanduser().resolve()
            for item in self.config["SCENES_DIR"].split(os.pathsep)
            if item.strip()
        ]
        return roots

    @staticmethod
    def _model_directory(scene_path: Path):
        for candidate in (scene_path / "sparse" / "0", scene_path / "sparse"):
            if candidate.is_dir():
                return candidate
        return None

    @staticmethod
    def _colmap_file_status(model_path):
        status = {}
        for name in (*COLMAP_CORE_FILES, *COLMAP_OPTIONAL_FILES):
            status[name] = (
                bool(
                    model_path
                    and any(
                        (model_path / f"{name}.{extension}").is_file()
                        for extension in ("bin", "txt")
                    )
                )
            )
        return status

    @staticmethod
    def _image_count(image_path):
        if not image_path or not image_path.is_dir():
            return 0
        try:
            return sum(
                item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
                for item in image_path.iterdir()
            )
        except OSError:
            return 0

    def _scene_record(
        self, scene_path: Path, model_override=None, ply_override=None, scene_name=None
    ):
        configured_model_path = self.config.get("COLMAP_MODEL_PATH")
        model_path = (
            Path(model_override).expanduser().resolve()
            if model_override
            else self._model_directory(scene_path)
        )
        if model_path is None and configured_model_path:
            model_path = Path(configured_model_path).expanduser().resolve()
        reference_override = self.config.get("REFERENCE_IMAGES_DIR")
        reference_path = None
        if reference_override:
            configured_reference_path = Path(reference_override).expanduser().resolve()
            if self._image_count(configured_reference_path):
                reference_path = configured_reference_path
        candidates = (
            scene_path / "images",
            scene_path / "data",
            scene_path / "images_2",
        )
        if reference_path is None:
            reference_path = next(
                (path for path in candidates if self._image_count(path)), None
            )
        if reference_path is None and reference_override:
            reference_path = configured_reference_path
        if reference_path is None:
            reference_path = next((path for path in candidates if path.is_dir()), None)
        query_path = Path(self.config["QUERY_IMAGE_PATH"]).expanduser().resolve() if self.config.get("QUERY_IMAGE_PATH") else None
        query_calib_path = (
            Path(self.config["QUERY_CALIB_PATH"]).expanduser().resolve()
            if self.config.get("QUERY_CALIB_PATH")
            else None
        )
        if ply_override:
            configured_ply = Path(ply_override).expanduser().resolve()
            ply_paths = [configured_ply] if configured_ply.is_file() else []
        else:
            ply_paths = sorted(scene_path.glob("*.ply")) if scene_path.is_dir() else []
        colmap_files = self._colmap_file_status(model_path)
        missing_core = [
            f"{name}.bin or {name}.txt"
            for name in COLMAP_CORE_FILES
            if not colmap_files[name]
        ]
        reference_count = self._image_count(reference_path)
        missing = list(missing_core)
        if reference_path is None:
            missing.append("reference image directory")
        elif not reference_path.is_dir():
            missing.append("reference image directory")
        elif reference_count == 0:
            missing.append("reference images")
        is_ready = not missing
        return {
            "id": _scene_id(Path(scene_name)) if scene_name else _scene_id(scene_path),
            "name": scene_name or scene_path.name,
            "path": str(scene_path),
            "model_path": str(model_path) if model_path else None,
            "reference_path": str(reference_path) if reference_path else None,
            "reference_image_count": reference_count,
            "colmap_files": colmap_files,
            "query_image_path": str(query_path) if query_path else None,
            "query_image_available": bool(query_path and query_path.is_file()),
            "query_calib_path": str(query_calib_path) if query_calib_path else None,
            "query_calib_available": bool(
                query_calib_path and query_calib_path.is_file()
            ),
            "ply_path": str(ply_paths[0]) if ply_paths else None,
            "has_gaussian_ply": bool(ply_paths),
            "ready": is_ready,
            "coordinate_frame_verified": bool(self.config["SCENE_FRAME_VERIFIED"]),
            "warning": None
            if is_ready
            else "Missing: " + ", ".join(missing),
        }

    def data_status(self):
        scenes = self.list_scenes()
        frame_verified = any(scene["coordinate_frame_verified"] for scene in scenes)
        return {
            "project_data_root": (
                str(Path(self.config["DATA_ROOT"]).expanduser().resolve())
                if self.config.get("DATA_ROOT")
                else None
            ),
            "scenes": [
                {
                    "id": scene["id"],
                    "name": scene["name"],
                    "ready": scene["ready"],
                    "ply_path": scene["ply_path"],
                    "ply_available": scene["has_gaussian_ply"],
                    "colmap_model_path": scene["model_path"],
                    "colmap_files": scene["colmap_files"],
                    "reference_images_path": scene["reference_path"],
                    "reference_image_count": scene["reference_image_count"],
                    "query_image_path": scene["query_image_path"],
                    "query_image_available": scene["query_image_available"],
                    "query_calib_path": scene["query_calib_path"],
                    "query_calib_available": scene["query_calib_available"],
                    "missing": scene["warning"],
                    "coordinate_frame_verified": scene["coordinate_frame_verified"],
                }
                for scene in scenes
            ],
            "notes": [
                (
                    "COLMAP 3D points and Gaussian PLY alignment is marked verified."
                    if frame_verified
                    else "COLMAP 3D points and Gaussian PLY alignment is unverified."
                ),
                "COLMAP rigs.bin and frames.bin are reported when present; the core sparse files are cameras, images, and points3D.",
            ],
        }

    def list_scenes(self):
        scenes = {}
        configured_model = self.config.get("COLMAP_MODEL_PATH")
        configured_scene_id = None
        if configured_model and Path(configured_model).expanduser().is_dir():
            model_path = Path(configured_model).expanduser().resolve()
            if model_path.parent.name == "sparse":
                scene_path = model_path.parent.parent
            elif model_path.name == "sparse":
                scene_path = model_path.parent
            else:
                scene_path = model_path.parent
            record = self._scene_record(
                scene_path,
                model_override=model_path,
                ply_override=self.config.get("GAUSSIAN_PLY_PATH") or None,
                scene_name=(
                    Path(self.config["GAUSSIAN_PLY_PATH"]).stem
                    if self.config.get("GAUSSIAN_PLY_PATH")
                    else None
                ),
            )
            scenes[record["id"]] = record
            configured_scene_id = record["id"]
        for root in self._roots():
            if not root.is_dir():
                continue
            if self._model_directory(root) or list(root.glob("*.ply")):
                record = self._scene_record(
                    root,
                )
                if record["id"] != configured_scene_id:
                    scenes[record["id"]] = record
            try:
                children = sorted(
                    (entry for entry in root.iterdir() if entry.is_dir()),
                    key=lambda entry: entry.name.casefold(),
                )
            except OSError:
                continue
            for child in children:
                if self._model_directory(child) or list(child.glob("*.ply")):
                    record = self._scene_record(child)
                    if record["id"] != configured_scene_id:
                        scenes[record["id"]] = record
        configured_ply = self.config.get("GAUSSIAN_PLY_PATH")
        if configured_ply and Path(configured_ply).expanduser().is_file():
            resolved_ply = Path(configured_ply).expanduser().resolve()
            if not any(scene["ply_path"] == str(resolved_ply) for scene in scenes.values()):
                record = self._scene_record(
                    resolved_ply.parent,
                    ply_override=resolved_ply,
                    scene_name=resolved_ply.stem,
                )
                scenes[record["id"]] = record
        return list(scenes.values())

    def get_scene(self, scene_id):
        for scene in self.list_scenes():
            if scene["id"] == scene_id:
                return scene
        return None
