import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import numpy as np
from app.config import project_data_paths
from app import create_app
from app.extensions import db
from app.localization.engine import LocalizationEngine, LocalizationUnavailable
from app.services.scenes import SceneManager


class FakeLocalizationEngine:
    def __init__(self):
        self.last_image_path = None

    def localize(self, image_path, scene):
        self.last_image_path = Path(image_path)
        return {
            "status": "localized",
            "message": None,
            "pose": {
                "coordinate_frame": "COLMAP_world",
                "translation_units": "COLMAP_scene_units_not_necessarily_meters",
                "pose_convention": "world_to_camera",
                "world_to_camera": [
                    [1, 0, 0, 1],
                    [0, 1, 0, 2],
                    [0, 0, 1, 3],
                    [0, 0, 0, 1],
                ],
                "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "translation": [1, 2, 3],
                "camera_center": [-1, -2, -3],
                "quaternion_wxyz": [1, 0, 0, 0],
                "euler_xyz_degrees": [0, 0, 0],
            },
            "metrics": {
                "inliers": 22,
                "inlier_ratio": 0.4,
                "median_reprojection_error_px": 1.2,
            },
        }


class UnavailableLocalizationEngine:
    def localize(self, image_path, scene):
        raise LocalizationUnavailable("COLMAP model is not configured.")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        scene = root / "scenes" / "road"
        (scene / "sparse" / "0").mkdir(parents=True)
        (scene / "images").mkdir()
        (scene / "images" / "reference.jpg").write_bytes(self.image_bytes())
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            (scene / "sparse" / "0" / name).write_bytes(b"fixture")
        self.app = create_app(
            {
                "TESTING": True,
                "SQLALCHEMY_DATABASE_URI": "sqlite://",
                "UPLOAD_DIR": root / "uploads",
                "SCENES_DIR": str(root / "scenes"),
                "DATA_ROOT": str(root / "external-data"),
                "COLMAP_MODEL_PATH": "",
                "REFERENCE_IMAGES_DIR": "",
                "QUERY_IMAGE_PATH": "",
                "QUERY_CALIB_PATH": "",
                "GAUSSIAN_PLY_PATH": "",
                "SCENE_FRAME_VERIFIED": False,
                "LLM_API_KEY": "",
            }
        )
        self.app.extensions["localization_engine"] = FakeLocalizationEngine()
        self.client = self.app.test_client()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()
            db.engine.dispose()
        self.temporary.cleanup()

    @staticmethod
    def image_bytes():
        image = Image.new("RGB", (96, 64), color=(72, 112, 145))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()

    def test_health_and_empty_history(self):
        self.assertEqual(self.client.get("/api/health").status_code, 200)
        self.assertEqual(
            self.client.get("/api/localizations").json["localizations"], []
        )

    def test_scene_discovery_is_data_driven(self):
        response = self.client.get("/api/scenes")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["scenes"][0]["id"], "road")
        self.assertTrue(response.json["scenes"][0]["ready"])

    def test_viewer_camera_uses_registered_colmap_poses(self):
        class FakeImage:
            camera_id = 1

            def __init__(self, x):
                self._matrix = np.eye(4)
                self._matrix[0, 3] = -x

            def cam_from_world(self):
                return self

            def matrix(self):
                return self._matrix

        class FakeCamera:
            height = 375
            focal_length_y = 300

        reconstruction = type(
            "FakeReconstruction",
            (),
            {
                "images": {1: FakeImage(0), 2: FakeImage(2)},
                "cameras": {1: FakeCamera()},
            },
        )()
        with patch(
            "app.routes.api.load_reconstruction", return_value=reconstruction
        ):
            response = self.client.get("/api/scenes/road/viewer-camera")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["registered_camera_count"], 2)
        self.assertEqual(
            response.json["source"],
            "registered COLMAP camera nearest the trajectory median",
        )
        self.assertEqual(response.json["position"], [0.0, 0.0, 0.0])
        self.assertEqual(response.json["target"], [0.0, 0.0, 8.0])
        self.assertAlmostEqual(response.json["fov"], 64.0108, places=3)

    def test_project_data_root_derives_drive_dataset_paths(self):
        paths = project_data_paths(r"G:\My Drive")
        self.assertEqual(
            Path(paths["colmap_model_path"]),
            Path(r"G:\My Drive\kitprojet\3DGS\sparse\0"),
        )
        self.assertEqual(
            Path(paths["reference_images_dir"]),
            Path(r"G:\My Drive\kitprojet\3DGS\data"),
        )
        self.assertEqual(
            Path(paths["query_image_path"]),
            Path(r"G:\My Drive\kitprojet\data\query.jpg"),
        )
        self.assertEqual(
            Path(paths["query_calib_path"]),
            Path(r"G:\My Drive\kitprojet\data\calib.txt"),
        )

    def test_data_status_reports_expected_files_without_fake_data(self):
        response = self.client.get("/api/data-status")
        self.assertEqual(response.status_code, 200)
        scene = response.json["scenes"][0]
        self.assertFalse(scene["ply_available"])
        self.assertFalse(scene["colmap_files"]["rigs"])
        self.assertEqual(scene["reference_image_count"], 1)
        self.assertFalse(scene["query_image_available"])
        self.assertIsNone(scene["missing"])

    def test_data_status_reflects_configured_frame_verification(self):
        self.app.config["SCENE_FRAME_VERIFIED"] = True
        response = self.client.get("/api/data-status")
        self.assertTrue(response.json["scenes"][0]["coordinate_frame_verified"])
        self.assertIn("marked verified", response.json["notes"][0])

    def test_configured_query_requires_a_real_query_file(self):
        response = self.client.post(
            "/api/localize/configured-query", json={"scene_id": "road"}
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("unavailable", response.json["error"])

    def test_configured_query_localizes_the_actual_configured_file(self):
        query_path = Path(self.temporary.name) / "query.jpg"
        query_path.write_bytes(self.image_bytes())
        self.app.config["QUERY_IMAGE_PATH"] = str(query_path)
        response = self.client.post(
            "/api/localize/configured-query", json={"scene_id": "road"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["localization"]["image_name"], "query.jpg")
        self.assertEqual(
            self.app.extensions["localization_engine"].last_image_path.resolve(),
            query_path.resolve(),
        )

    def test_gaussian_ply_is_listed_without_claiming_localization_readiness(self):
        root = Path(self.temporary.name)
        ply = root / "outside" / "road scene.ply"
        ply.parent.mkdir()
        ply.write_text("ply\n", encoding="ascii")
        manager = SceneManager(
            {
                "SCENES_DIR": str(root / "empty-scenes"),
                "COLMAP_MODEL_PATH": "",
                "GAUSSIAN_PLY_PATH": str(ply),
                "REFERENCE_IMAGES_DIR": "",
                "SCENE_FRAME_VERIFIED": False,
            }
        )
        scene = manager.list_scenes()[0]
        self.assertEqual(scene["name"], "road scene")
        self.assertTrue(scene["has_gaussian_ply"])
        self.assertFalse(scene["ready"])

    def test_empty_external_data_root_is_not_mistaken_for_reference_images(self):
        root = Path(self.temporary.name)
        project = root / "project"
        project.mkdir()
        (project / "data").mkdir()
        ply = project / "road.ply"
        ply.write_text("ply\n", encoding="ascii")
        expected_references = root / "data" / "kitprojet" / "3DGS" / "data"
        manager = SceneManager(
            {
                "SCENES_DIR": str(root / "empty-scenes"),
                "COLMAP_MODEL_PATH": "",
                "GAUSSIAN_PLY_PATH": str(ply),
                "REFERENCE_IMAGES_DIR": str(expected_references),
                "SCENE_FRAME_VERIFIED": False,
            }
        )
        scene = manager.list_scenes()[0]
        self.assertEqual(Path(scene["reference_path"]), expected_references.resolve())
        self.assertEqual(scene["reference_image_count"], 0)

    def test_localization_persists_pose_and_visualization(self):
        response = self.client.post(
            "/api/localize",
            data={
                "scene_id": "road",
                "image": (io.BytesIO(self.image_bytes()), "query.png"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        item = response.json["localization"]
        self.assertEqual(item["status"], "localized")
        self.assertEqual(item["metrics"]["inliers"], 22)
        detail = self.client.get(f"/api/localizations/{item['id']}")
        self.assertEqual(detail.json["localization"]["pose"]["camera_center"], [-1, -2, -3])
        visualization = self.client.get(
            f"/api/localizations/{item['id']}/visualization"
        )
        self.assertEqual(visualization.status_code, 200)
        self.assertFalse(visualization.json["overlay_allowed"])

    def test_invalid_upload_is_rejected(self):
        response = self.client.post(
            "/api/localize",
            data={
                "scene_id": "road",
                "image": (io.BytesIO(b"not an image"), "query.png"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("valide", response.json["error"])

    def test_featureless_upload_returns_plain_language_failure_without_loading_scene(self):
        self.app.extensions["localization_engine"] = LocalizationEngine(
            self.app.config
        )
        response = self.client.post(
            "/api/localize",
            data={
                "scene_id": "road",
                "image": (io.BytesIO(self.image_bytes()), "flat-image.png"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json["localization"]["status"], "failed")
        self.assertIn("aucun détail visible", response.json["error"].lower())

    def test_unavailable_engine_is_reported_and_saved_as_failed(self):
        self.app.extensions["localization_engine"] = UnavailableLocalizationEngine()
        response = self.client.post(
            "/api/localize",
            data={
                "scene_id": "road",
                "image": (io.BytesIO(self.image_bytes()), "query.png"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["localization"]["status"], "failed")
        self.assertEqual(
            response.json["localization"]["error"], "COLMAP model is not configured."
        )

    def test_copilot_fallback_is_explicit_when_unconfigured(self):
        response = self.client.post(
            "/api/copilot/chat",
            json={"messages": [{"role": "user", "content": "Explain the pipeline."}]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["provider_configured"])
        self.assertIn("not configured", response.json["reply"])

    def test_copilot_executes_localization_tool_after_explicit_request(self):
        service = self.app.extensions["copilot_service"]
        service.localization_engine = self.app.extensions["localization_engine"]
        self.app.config["LLM_API_KEY"] = "test-key"
        request_count = 0

        def fake_model_request(conversation, tools):
            nonlocal request_count
            request_count += 1
            tool_names = [tool["function"]["name"] for tool in tools]
            self.assertIn("run_localization", tool_names)
            if request_count == 1:
                return {
                    "tool_calls": [
                        {
                            "id": "test-call",
                            "function": {
                                "name": "run_localization",
                                "arguments": json.dumps({"scene_id": "road"}),
                            },
                        }
                    ]
                }
            tool_result = json.loads(conversation[-1]["content"])
            self.assertEqual(tool_result["localization"]["status"], "localized")
            return {"content": "Localization completed with the stored test result."}

        service._request_model = fake_model_request
        with self.app.app_context():
            response = service.chat(
                [{"role": "user", "content": "Please localize this image."}],
                (self.image_bytes(), "query.png"),
                "road",
            )
        self.assertTrue(response["provider_configured"])
        self.assertEqual(request_count, 2)


if __name__ == "__main__":
    unittest.main()
