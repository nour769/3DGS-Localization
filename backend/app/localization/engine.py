import logging
from pathlib import Path

import numpy as np

from .colmap import ColmapModelError, load_reconstruction
from .geometry import (
    camera_center_world,
    quaternion_to_euler_xyz_degrees,
    rotation_to_quaternion_wxyz,
    scaled_intrinsics,
)

logger = logging.getLogger(__name__)


class LocalizationError(RuntimeError):
    pass


class LocalizationUnavailable(LocalizationError):
    pass


class LocalizationEngine:
    """SIFT correspondences to observed COLMAP 3D points, followed by PnP-RANSAC."""

    def __init__(self, config):
        self.config = config
        self._indexes = {}
        self._dino = None
        self._torch = None

    def _cv2(self):
        try:
            import cv2
        except ImportError as exc:
            raise LocalizationUnavailable(
                "OpenCV is missing. Install backend/requirements.txt."
            ) from exc
        return cv2

    def _resize(self, image):
        cv2 = self._cv2()
        height, width = image.shape[:2]
        target_width = int(self.config["IMAGE_WORK_WIDTH"])
        target_height = max(1, round(height * target_width / width))
        resized = cv2.resize(
            image, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
        return resized, target_width / width, target_height / height

    def _load_dino(self):
        if not self.config["ENABLE_DINOV2"]:
            return None
        if self._dino is not None:
            return self._dino
        try:
            import torch
        except ImportError as exc:
            raise LocalizationUnavailable(
                "DINOv2 is enabled but PyTorch is not installed. "
                "Install backend/requirements-ml.txt or set ENABLE_DINOV2=0."
            ) from exc
        try:
            model = torch.hub.load(
                "facebookresearch/dinov2", self.config["DINO_MODEL"]
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise LocalizationUnavailable(
                f"Could not load DINOv2 model {self.config['DINO_MODEL']}: {exc}"
            ) from exc
        self._torch = torch
        self._dino = model.eval()
        return self._dino

    def _global_descriptor(self, rgb):
        model = self._load_dino()
        if model is None:
            return None
        cv2 = self._cv2()
        torch = self._torch
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float()[None] / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        with torch.no_grad():
            descriptor = model((tensor - mean) / std)
            return torch.nn.functional.normalize(descriptor, dim=-1)[0].cpu().numpy()

    def _histogram(self, rgb):
        cv2 = self._cv2()
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        histogram = cv2.calcHist(
            [hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]
        ).reshape(-1)
        norm = np.linalg.norm(histogram)
        return histogram / norm if norm else histogram

    @staticmethod
    def _pose_observations(image):
        points = []
        point_ids = []
        for observation in image.points2D:
            raw_point_id = observation.point3D_id
            if raw_point_id is None:
                continue
            point_id = int(raw_point_id)
            xy = np.asarray(observation.xy, dtype=np.float64).reshape(2)
            points.append(xy)
            point_ids.append(point_id)
        return np.asarray(points, dtype=np.float64), np.asarray(point_ids, dtype=np.uint64)

    def _build_reference(self, image, reconstruction, image_root):
        from scipy.spatial import cKDTree

        cv2 = self._cv2()
        path = (image_root / image.name).resolve()
        if not path.is_file():
            return None
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("Skipping unreadable COLMAP reference image: %s", path)
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized, scale_x, scale_y = self._resize(rgb)
        colmap_points, point_ids = self._pose_observations(image)
        valid = np.array(
            [int(point_id) in reconstruction.points3D for point_id in point_ids],
            dtype=bool,
        )
        if not valid.any():
            return None
        reference_points = colmap_points[valid] * [scale_x, scale_y]
        reference_ids = point_ids[valid]
        keypoints, descriptors = cv2.SIFT_create(nfeatures=5000).detectAndCompute(
            cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY), None
        )
        if descriptors is None or not keypoints:
            return None
        sift_points = np.asarray([keypoint.pt for keypoint in keypoints])
        distances, indices = cKDTree(reference_points).query(
            sift_points, distance_upper_bound=4.0
        )
        matched = np.isfinite(distances) & (indices < len(reference_ids))
        if int(matched.sum()) < 8:
            return None
        matched_descriptors = descriptors[matched]
        xyz = np.asarray(
            [
                reconstruction.points3D[int(point_id)].xyz
                for point_id in reference_ids[indices[matched]]
            ],
            dtype=np.float64,
        )
        return {
            "name": image.name,
            "descriptors": matched_descriptors,
            "xyz": xyz,
            "histogram": self._histogram(resized),
            "global": self._global_descriptor(resized),
        }

    def _load_index(self, scene):
        scene_id = scene["id"]
        if scene_id in self._indexes:
            return self._indexes[scene_id]
        if not scene["ready"]:
            raise LocalizationUnavailable(scene["warning"] or "Scene is not ready.")
        try:
            reconstruction = load_reconstruction(Path(scene["model_path"]))
        except ColmapModelError as exc:
            raise LocalizationUnavailable(str(exc)) from exc
        references = []
        for image in reconstruction.images.values():
            reference = self._build_reference(
                image, reconstruction, Path(scene["reference_path"])
            )
            if reference is not None:
                references.append(reference)
        if not references:
            raise LocalizationUnavailable(
                "No reference images with SIFT features linked to COLMAP 3D points "
                "were found. Check the image paths and reconstruction observations."
            )
        index = {"reconstruction": reconstruction, "references": references}
        self._indexes[scene_id] = index
        return index

    def _query_intrinsics(self, source_width, source_height, target_width, target_height):
        base_width = int(self.config["QUERY_WIDTH"])
        base_height = int(self.config["QUERY_HEIGHT"])
        if base_width <= 0 or base_height <= 0:
            raise LocalizationError("Query calibration dimensions must be positive.")
        calibration_path = self.config.get("QUERY_CALIB_PATH")
        if calibration_path and Path(calibration_path).is_file():
            try:
                values = np.asarray(
                    Path(calibration_path).read_text(encoding="utf-8").split(),
                    dtype=np.float64,
                )
            except (OSError, ValueError) as exc:
                raise LocalizationError(
                    f"Could not read query calibration at {calibration_path}: {exc}"
                ) from exc
            if values.shape != (4,):
                raise LocalizationError(
                    "Query calibration file must contain exactly four values: fx fy cx cy."
                )
            calibration = values
        else:
            calibration = np.asarray(
                [
                    self.config["QUERY_FX"],
                    self.config["QUERY_FY"],
                    self.config["QUERY_CX"],
                    self.config["QUERY_CY"],
                ],
                dtype=np.float64,
            )
        if not np.isfinite(calibration).all() or min(calibration[:2]) <= 0:
            raise LocalizationError(
                "Query focal lengths and principal point must be finite; focal lengths must be positive."
            )
        fx = calibration[0] * source_width / base_width
        fy = calibration[1] * source_height / base_height
        cx = calibration[2] * source_width / base_width
        cy = calibration[3] * source_height / base_height
        return scaled_intrinsics(
            fx,
            fy,
            cx,
            cy,
            source_width,
            source_height,
            target_width,
            target_height,
        )

    def _select_references(self, references, query_rgb, top_k=10):
        query_descriptor = self._global_descriptor(query_rgb)
        if query_descriptor is not None:
            scores = np.asarray(
                [
                    float(np.dot(reference["global"], query_descriptor))
                    for reference in references
                ]
            )
            method = "dinov2"
        else:
            query_histogram = self._histogram(query_rgb)
            scores = np.asarray(
                [
                    float(np.dot(reference["histogram"], query_histogram))
                    for reference in references
                ]
            )
            method = "color_histogram_fallback"
        indices = np.argsort(scores)[::-1][: min(top_k, len(references))]
        return [(references[int(index)], float(scores[index])) for index in indices], method

    def _estimate(
        self, query_keypoints, query_descriptors, references, camera_matrix
    ):
        cv2 = self._cv2()
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        candidates = []
        for reference, score in references:
            matches = matcher.knnMatch(query_descriptors, reference["descriptors"], k=2)
            good = [
                first
                for pair in matches
                if len(pair) == 2
                for first, second in [pair]
                if first.distance < 0.78 * second.distance
            ]
            if len(good) < 6:
                continue
            query_points = np.asarray(
                [query_keypoints[match.queryIdx].pt for match in good],
                dtype=np.float64,
            )
            world_points = np.asarray(
                [reference["xyz"][match.trainIdx] for match in good],
                dtype=np.float64,
            )
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                world_points,
                query_points,
                camera_matrix,
                None,
                iterationsCount=3000,
                reprojectionError=float(self.config["PNP_REPROJECTION_ERROR"]),
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not success or inliers is None or len(inliers) < 4:
                continue
            indices = inliers.reshape(-1)
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    world_points[indices],
                    query_points[indices],
                    camera_matrix,
                    None,
                    rvec,
                    tvec,
                )
            except cv2.error:
                logger.info("PnP refinement did not converge; retaining RANSAC pose.")
            rotation, _ = cv2.Rodrigues(rvec)
            projected, _ = cv2.projectPoints(
                world_points[indices], rvec, tvec, camera_matrix, None
            )
            errors = np.linalg.norm(
                projected.reshape(-1, 2) - query_points[indices], axis=1
            )
            candidates.append(
                {
                    "rotation": rotation,
                    "translation": tvec.reshape(3),
                    "inliers": int(len(indices)),
                    "inlier_ratio": float(len(indices) / len(good)),
                    "mean_reprojection_error_px": float(np.mean(errors)),
                    "median_reprojection_error_px": float(np.median(errors)),
                    "reference_name": reference["name"],
                    "reference_score": score,
                    "matches": len(good),
                }
            )
        return max(
            candidates,
            key=lambda result: (
                result["inliers"],
                -result["median_reprojection_error_px"],
                result["reference_score"],
            ),
            default=None,
        )

    def localize(self, image_path, scene):
        cv2 = self._cv2()
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise LocalizationError("Uploaded file is not a readable image.")
        source_height, source_width = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        query, _, _ = self._resize(rgb)
        query_gray = cv2.cvtColor(query, cv2.COLOR_RGB2GRAY)
        query_keypoints, query_descriptors = cv2.SIFT_create(nfeatures=5000).detectAndCompute(
            query_gray, None
        )
        if query_descriptors is None or len(query_keypoints) < 8:
            raise LocalizationError("Too few SIFT features were detected in the query image.")
        index = self._load_index(scene)
        selected, retrieval_method = self._select_references(
            index["references"], query
        )
        camera_matrix = self._query_intrinsics(
            source_width,
            source_height,
            query.shape[1],
            query.shape[0],
        )
        result = self._estimate(
            query_keypoints, query_descriptors, selected, camera_matrix
        )
        if result is None:
            return {
                "status": "failed",
                "message": "PnP-RANSAC could not estimate a pose from the selected references.",
                "pose": None,
                "metrics": {
                    "retrieval_method": retrieval_method,
                    "candidate_references": [ref["name"] for ref, _ in selected],
                    "query_size": [source_width, source_height],
                    "working_size": [query.shape[1], query.shape[0]],
                    "intrinsics": camera_matrix.tolist(),
                },
            }
        rotation = result.pop("rotation")
        translation = result.pop("translation")
        center = camera_center_world(rotation, translation)
        quaternion = rotation_to_quaternion_wxyz(rotation)
        euler = quaternion_to_euler_xyz_degrees(quaternion)
        minimum_inliers = int(self.config["MIN_PNP_INLIERS"])
        is_localized = (
            result["inliers"] >= minimum_inliers
            and result["median_reprojection_error_px"]
            <= float(self.config["PNP_REPROJECTION_ERROR"])
            and result["inlier_ratio"] >= 0.1
        )
        pose = {
            "coordinate_frame": "COLMAP_world",
            "translation_units": "COLMAP_scene_units_not_necessarily_meters",
            "pose_convention": "world_to_camera",
            "world_to_camera": np.column_stack(
                [rotation, translation]
            ).tolist()
            + [[0.0, 0.0, 0.0, 1.0]],
            "rotation_matrix": rotation.tolist(),
            "translation": translation.tolist(),
            "camera_center": center.tolist(),
            "quaternion_wxyz": quaternion.tolist(),
            "euler_xyz_degrees": euler.tolist(),
        }
        metrics = {
            **result,
            "retrieval_method": retrieval_method,
            "candidate_references": [ref["name"] for ref, _ in selected],
            "query_size": [source_width, source_height],
            "working_size": [query.shape[1], query.shape[0]],
            "intrinsics": camera_matrix.tolist(),
            "intrinsics_scaled_from": [
                int(self.config["QUERY_WIDTH"]),
                int(self.config["QUERY_HEIGHT"]),
            ],
            "calibration_warning": self._calibration_warning(
                source_width, source_height
            ),
            "coordinate_scale_verified_metric": False,
            "gaussian_colmap_alignment_verified": bool(
                scene["coordinate_frame_verified"]
            ),
        }
        return {
            "status": "localized" if is_localized else "uncertain",
            "message": None
            if is_localized
            else "A pose was estimated, but it did not meet all confidence thresholds.",
            "pose": pose,
            "metrics": metrics,
        }

    def _calibration_warning(self, image_width, image_height):
        image_ratio = image_width / image_height
        calibration_ratio = (
            int(self.config["QUERY_WIDTH"]) / int(self.config["QUERY_HEIGHT"])
        )
        if abs(image_ratio / calibration_ratio - 1.0) > 0.01:
            return (
                "Query aspect ratio differs from the configured calibration by more "
                "than 1%; verify cropping or camera calibration."
            )
        return None
