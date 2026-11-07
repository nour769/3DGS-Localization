import math
from pathlib import Path

import numpy as np
from flask import Blueprint, current_app, jsonify, request, send_file
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ..extensions import db
from ..localization.colmap import ColmapModelError, load_reconstruction
from ..localization.engine import LocalizationError, LocalizationUnavailable
from ..models import Localization
from ..services.copilot import CopilotError
from ..services.localization import (
    create_record,
    execute_localization,
    save_upload,
    validate_image,
)

api = Blueprint("api", __name__, url_prefix="/api")


def _scene_manager():
    return current_app.extensions["scene_manager"]


def _public_scene(scene):
    public = {
        key: scene[key]
        for key in (
            "id",
            "name",
            "ready",
            "has_gaussian_ply",
            "coordinate_frame_verified",
            "warning",
        )
    }
    public["model_url"] = (
        f"/api/scenes/{scene['id']}/model" if scene["ply_path"] else None
    )
    return public


@api.get("/health")
def health():
    try:
        db.session.execute(text("SELECT 1"))
    except SQLAlchemyError:
        current_app.logger.exception("Database health check failed.")
        return jsonify({"status": "degraded", "database": "unavailable"}), 503
    return jsonify({"status": "ok", "database": "ok"})


@api.get("/scenes")
def scenes():
    return jsonify({"scenes": [_public_scene(scene) for scene in _scene_manager().list_scenes()]})

@api.get("/data-status")
def data_status():
    return jsonify(_scene_manager().data_status())


@api.get("/scenes/<scene_id>/model")
def scene_model(scene_id):
    scene = _scene_manager().get_scene(scene_id)
    if scene is None or not scene["ply_path"]:
        return jsonify({"error": "No Gaussian PLY model is configured for this scene."}), 404
    return send_file(
        Path(scene["ply_path"]),
        mimetype="application/octet-stream",
        conditional=True,
        download_name=Path(scene["ply_path"]).name,
    )


@api.get("/scenes/<scene_id>/viewer-camera")
def scene_viewer_camera(scene_id):
    scene = _scene_manager().get_scene(scene_id)
    if scene is None or not scene["model_path"]:
        return jsonify({"error": "A COLMAP model is required to frame this scene."}), 404
    try:
        reconstruction = load_reconstruction(Path(scene["model_path"]))
    except (ColmapModelError, OSError) as exc:
        return jsonify({"error": str(exc)}), 503

    centers = []
    viewing_directions = []
    registered_images = []
    for image in reconstruction.images.values():
        world_to_camera = image.cam_from_world().matrix()
        rotation = np.asarray(world_to_camera[:3, :3], dtype=np.float64)
        translation = np.asarray(world_to_camera[:3, 3], dtype=np.float64)
        centers.append(-rotation.T @ translation)
        viewing_directions.append(rotation.T @ np.array([0.0, 0.0, 1.0]))
        registered_images.append(image)
    if not centers:
        return jsonify({"error": "The COLMAP model contains no registered camera poses."}), 503

    center_array = np.asarray(centers)
    camera_position_median = np.median(center_array, axis=0)
    closest_index = int(
        np.argmin(np.linalg.norm(center_array - camera_position_median, axis=1))
    )
    camera_position = center_array[closest_index]
    direction = viewing_directions[closest_index]
    direction_norm = float(np.linalg.norm(direction))
    direction /= direction_norm
    camera = reconstruction.cameras[registered_images[closest_index].camera_id]
    fov = math.degrees(
        2.0 * math.atan(camera.height / (2.0 * camera.focal_length_y))
    )
    trajectory_extent = float(np.linalg.norm(np.ptp(center_array, axis=0)))
    target_distance = max(trajectory_extent * 0.6, 8.0)
    target = camera_position + direction * target_distance
    return jsonify(
        {
            "position": camera_position.tolist(),
            "target": target.tolist(),
            "fov": fov,
            "registered_camera_count": len(centers),
            "source": "registered COLMAP camera nearest the trajectory median",
        }
    )


@api.post("/localize")
def localize():
    upload = request.files.get("image")
    scene_id = request.form.get("scene_id", "").strip()
    if upload is None:
        return jsonify({"error": "An image file is required."}), 400
    if not scene_id:
        return jsonify({"error": "scene_id is required."}), 400
    scene = _scene_manager().get_scene(scene_id)
    if scene is None:
        return jsonify({"error": "The selected scene is not configured."}), 404
    image_bytes = upload.read()
    try:
        validate_image(image_bytes)
        image_path = save_upload(
            image_bytes, upload.filename or "query.png", current_app.config["UPLOAD_DIR"]
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    record = create_record(upload.filename or "query-image", image_path, scene_id)
    try:
        result = execute_localization(
            record, scene, current_app.extensions["localization_engine"]
        )
        return jsonify({"localization": record.to_dict(), "result": result})
    except LocalizationUnavailable as exc:
        current_app.logger.warning("Localization unavailable: %s", exc)
        return jsonify({"localization": record.to_dict(), "error": str(exc)}), 503
    except LocalizationError as exc:
        current_app.logger.info("Localization could not be completed: %s", exc)
        return jsonify({"localization": record.to_dict(), "error": str(exc)}), 422


@api.post("/localize/configured-query")
def localize_configured_query():
    payload = request.get_json(silent=True) or {}
    scene_id = str(payload.get("scene_id", "")).strip()
    if not scene_id:
        return jsonify({"error": "scene_id is required."}), 400
    scene = _scene_manager().get_scene(scene_id)
    if scene is None:
        return jsonify({"error": "The selected scene is not configured."}), 404
    query_path = scene.get("query_image_path")
    if not query_path or not Path(query_path).is_file():
        return jsonify(
            {
                "error": (
                    "Configured query image is unavailable. Set QUERY_IMAGE_PATH or "
                    "PROJECT_DATA_ROOT to a locally accessible Google Drive MyDrive folder."
                )
            }
        ), 503
    query_path = Path(query_path)
    try:
        validate_image(query_path.read_bytes())
    except (OSError, ValueError) as exc:
        return jsonify({"error": f"Configured query image is invalid: {exc}"}), 422
    record = create_record(query_path.name, query_path, scene_id)
    try:
        result = execute_localization(
            record, scene, current_app.extensions["localization_engine"]
        )
        return jsonify({"localization": record.to_dict(), "result": result})
    except LocalizationUnavailable as exc:
        current_app.logger.warning("Configured-query localization unavailable: %s", exc)
        return jsonify({"localization": record.to_dict(), "error": str(exc)}), 503
    except LocalizationError as exc:
        current_app.logger.info("Configured-query localization failed: %s", exc)
        return jsonify({"localization": record.to_dict(), "error": str(exc)}), 422


@api.get("/localizations")
def localizations():
    try:
        limit = max(1, min(int(request.args.get("limit", "100")), 500))
    except ValueError:
        return jsonify({"error": "limit must be an integer."}), 400
    items = Localization.query.order_by(Localization.created_at.desc()).limit(limit).all()
    return jsonify({"localizations": [item.to_dict() for item in items]})


@api.get("/localizations/<localization_id>")
def localization_detail(localization_id):
    item = db.session.get(Localization, localization_id)
    if item is None:
        return jsonify({"error": "Localization not found."}), 404
    return jsonify({"localization": item.to_dict()})


@api.get("/localizations/<localization_id>/visualization")
def localization_visualization(localization_id):
    item = db.session.get(Localization, localization_id)
    if item is None:
        return jsonify({"error": "Localization not found."}), 404
    if item.pose_json is None:
        return jsonify({"error": "This localization does not contain an estimated pose."}), 409
    scene = _scene_manager().get_scene(item.scene_id)
    return jsonify(
        {
            "localization_id": item.id,
            "scene_id": item.scene_id,
            "pose": item.pose_json,
            "metrics": item.metrics_json or {},
            "scene": _public_scene(scene) if scene else None,
            "overlay_allowed": bool(scene and scene["coordinate_frame_verified"]),
            "coordinate_frame_warning": None
            if scene and scene["coordinate_frame_verified"]
            else (
                "The Gaussian PLY and COLMAP coordinate-frame alignment has not been "
                "verified. Camera overlays are disabled to avoid implying a validated alignment."
            ),
        }
    )


@api.post("/copilot/chat")
def copilot_chat():
    if request.mimetype == "multipart/form-data":
        messages_raw = request.form.get("messages", "[]")
        upload = request.files.get("image")
        attached_image = (upload.read(), upload.filename or "query.png") if upload else None
        scene_id = request.form.get("scene_id")
    else:
        payload = request.get_json(silent=True) or {}
        messages_raw = payload.get("messages", [])
        attached_image = None
        scene_id = payload.get("scene_id")
    if isinstance(messages_raw, str):
        import json

        try:
            messages = json.loads(messages_raw)
        except json.JSONDecodeError:
            return jsonify({"error": "messages must be a JSON array."}), 400
    else:
        messages = messages_raw
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(item, dict)
            or item.get("role") not in {"user", "assistant"}
            or not isinstance(item.get("content"), str)
            for item in messages
        )
    ):
        return jsonify({"error": "Provide at least one valid user/assistant message."}), 400
    if attached_image:
        try:
            validate_image(attached_image[0])
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    try:
        result = current_app.extensions["copilot_service"].chat(
            messages, attached_image, scene_id
        )
    except CopilotError as exc:
        current_app.logger.error("Copilot request failed: %s", exc)
        return jsonify({"error": str(exc)}), 502
    return jsonify(result)
