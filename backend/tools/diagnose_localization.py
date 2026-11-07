import csv
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app


PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data" / "kitprojet"
QUERY_PATH = DATA / "data" / "query.jpg"
REFERENCE_ROOT = DATA / "3DGS" / "data"
DEPTH_PATH = PROJECT / "rendu" / "data" / "depth.npy"
PLY_PATH = PROJECT / "scene-routiere-finale (2).ply"
OUTPUT = PROJECT / "diagnostics" / "query_localization_20261107"
KNOWN_REFERENCES = (
    "0000000000.png",
    "0000000111.png",
    "0000000223.png",
    "0000000334.png",
    "0000000446.png",
)
PNP_THRESHOLD = 4.0
PNP_CONFIDENCE = 0.999
PNP_ITERATIONS = 3000
LOWE_RATIO = 0.78


def _read_image(path, cv):
    bgr = cv.imread(str(path), cv.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Image is unreadable: {path}")
    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB), bgr


def _features_for_reference(engine, reconstruction, reference, sift, cv):
    path = REFERENCE_ROOT / reference["name"]
    rgb, _ = _read_image(path, cv)
    working, scale_x, scale_y = engine._resize(rgb)
    keypoints, descriptors = sift.detectAndCompute(
        cv.cvtColor(working, cv.COLOR_RGB2GRAY), None
    )
    image = next(
        item for item in reconstruction.images.values() if item.name == reference["name"]
    )
    colmap_points, point_ids = engine._pose_observations(image)
    valid = np.array(
        [int(point_id) in reconstruction.points3D for point_id in point_ids],
        dtype=bool,
    )
    reference_points = colmap_points[valid] * [scale_x, scale_y]
    reference_ids = point_ids[valid]
    sift_points = np.asarray([point.pt for point in keypoints])
    distances, indices = cKDTree(reference_points).query(
        sift_points, distance_upper_bound=4.0
    )
    mapped = np.isfinite(distances) & (indices < len(reference_ids))
    mapped_keypoints = [keypoints[index] for index in np.flatnonzero(mapped)]
    return working, keypoints, mapped_keypoints, descriptors


def _run_pnp(engine, query_keypoints, query_descriptors, reference, camera_matrix, seed):
    cv = cv2
    matcher = cv.BFMatcher(cv.NORM_L2)
    pairs = matcher.knnMatch(query_descriptors, reference["descriptors"], k=2)
    good = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < LOWE_RATIO * second.distance
    ]
    result = {
        "raw_knn_pairs": len(pairs),
        "pairs_with_2_neighbors": sum(len(pair) == 2 for pair in pairs),
        "after_lowe_ratio_0_78": len(good),
        "pnp_correspondences": len(good),
        "ransac_success": False,
        "ransac_inliers": None,
        "outliers": None,
    }
    if len(good) < 6:
        return result, None

    query_points = np.asarray(
        [query_keypoints[match.queryIdx].pt for match in good], dtype=np.float64
    )
    world_points = np.asarray(
        [reference["xyz"][match.trainIdx] for match in good], dtype=np.float64
    )
    cv.setRNGSeed(seed)
    success, rvec, tvec, inliers = cv.solvePnPRansac(
        world_points,
        query_points,
        camera_matrix,
        None,
        iterationsCount=PNP_ITERATIONS,
        reprojectionError=PNP_THRESHOLD,
        confidence=PNP_CONFIDENCE,
        flags=cv.SOLVEPNP_EPNP,
    )
    if not success or inliers is None:
        return result, None

    inlier_indices = inliers.reshape(-1)
    rvec, tvec = cv.solvePnPRefineLM(
        world_points[inlier_indices],
        query_points[inlier_indices],
        camera_matrix,
        None,
        rvec,
        tvec,
    )
    rotation, _ = cv.Rodrigues(rvec)
    projected, _ = cv.projectPoints(
        world_points, rvec, tvec, camera_matrix, None
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - query_points, axis=1)
    inlier_mask = np.zeros(len(good), dtype=bool)
    inlier_mask[inlier_indices] = True
    result.update(
        ransac_success=True,
        ransac_inliers=int(inlier_mask.sum()),
        outliers=int((~inlier_mask).sum()),
        median_inlier_error_px=float(np.median(errors[inlier_mask])),
        median_outlier_error_px=(
            float(np.median(errors[~inlier_mask])) if (~inlier_mask).any() else None
        ),
        median_all_error_px=float(np.median(errors)),
    )
    return result, {
        "query_points": query_points,
        "world_points": world_points,
        "good": good,
        "inlier_mask": inlier_mask,
        "errors": errors,
        "rotation": rotation,
        "translation": tvec.reshape(3),
    }


def _load_ply_xyz(path):
    ply_types = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "<i2",
        "int16": "<i2",
        "ushort": "<u2",
        "uint16": "<u2",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise RuntimeError(f"Malformed PLY header: {path}")
            text = line.decode("ascii").strip()
            header.append(text)
            if text == "end_header":
                break
        if not any(line.startswith("format binary_little_endian") for line in header):
            raise RuntimeError("Only binary little-endian PLY is supported by this report.")
        vertex_line = next(i for i, line in enumerate(header) if line.startswith("element vertex "))
        count = int(header[vertex_line].split()[2])
        properties = []
        for line in header[vertex_line + 1 :]:
            if not line.startswith("property "):
                break
            _, type_name, name = line.split()
            properties.append((name, ply_types[type_name]))
        if not {"x", "y", "z"}.issubset({name for name, _ in properties}):
            raise RuntimeError("PLY vertex does not contain xyz properties.")
        dtype = np.dtype(properties, align=False)
        vertices = np.fromfile(stream, dtype=dtype, count=count)
    return np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(
        np.float64
    )


def _get_latest_query_record():
    with urllib.request.urlopen(
        "http://127.0.0.1:5000/api/localizations?limit=100", timeout=30
    ) as response:
        history = json.load(response)["localizations"]
    record = next(
        (
            item
            for item in history
            if item.get("image_name") == "query.jpg"
            and item.get("pose")
            and item.get("scene_id") == "scene-routiere-finale-2"
        ),
        None,
    )
    if record is None:
        raise RuntimeError("No real query.jpg pose is persisted in the running Flask API.")
    with urllib.request.urlopen(
        f"http://127.0.0.1:5000/api/localizations/{record['id']}", timeout=30
    ) as response:
        return json.load(response)["localization"]


def _cross_check_references(engine, reconstruction, references, sift, cv):
    rows = []
    visual_results = {}
    matcher = cv.BFMatcher(cv.NORM_L2)
    for query_index, name in enumerate(KNOWN_REFERENCES):
        image = next(item for item in reconstruction.images.values() if item.name == name)
        camera = reconstruction.cameras[image.camera_id]
        rgb, _ = _read_image(REFERENCE_ROOT / name, cv)
        working, scale_x, scale_y = engine._resize(rgb)
        keypoints, descriptors = sift.detectAndCompute(
            cv.cvtColor(working, cv.COLOR_RGB2GRAY), None
        )
        source_intrinsics = np.asarray(camera.calibration_matrix(), dtype=np.float64)
        working_intrinsics = source_intrinsics.copy()
        working_intrinsics[0, :] *= scale_x
        working_intrinsics[1, :] *= scale_y

        candidates = [reference for reference in references if reference["name"] != name]
        selected, method = engine._select_references(candidates, working)
        best = None
        for rank, (reference, score) in enumerate(selected, 1):
            match, result = _run_pnp(
                engine,
                keypoints,
                descriptors,
                reference,
                working_intrinsics,
                20261107 + 1000 * query_index + rank,
            )
            if result is None:
                continue
            ground_truth = image.cam_from_world().matrix()
            rotation_gt = ground_truth[:3, :3]
            center_gt = -rotation_gt.T @ ground_truth[:3, 3]
            center_est = -result["rotation"].T @ result["translation"]
            cosine = np.clip(
                (np.trace(result["rotation"] @ rotation_gt.T) - 1) / 2, -1, 1
            )
            row = {
                "query_reference": name,
                "camera_id": int(image.camera_id),
                "camera_model": str(camera.model),
                "camera_size": [int(camera.width), int(camera.height)],
                "source_intrinsics": source_intrinsics.tolist(),
                "working_intrinsics": working_intrinsics.tolist(),
                "working_size": [int(working.shape[1]), int(working.shape[0])],
                "query_sift_keypoints": len(keypoints),
                "retrieval_method": method,
                "self_reference_excluded": True,
                "best_reference": reference["name"],
                "retrieval_rank": rank,
                "similarity_score": float(score),
                "pnp_correspondences": match["pnp_correspondences"],
                "inliers": match["ransac_inliers"],
                "inlier_ratio": match["ransac_inliers"] / match["pnp_correspondences"],
                "median_inlier_reprojection_px": match["median_inlier_error_px"],
                "camera_center_error_colmap_units": float(
                    np.linalg.norm(center_est - center_gt)
                ),
                "rotation_error_degrees": float(
                    np.degrees(np.arccos(cosine))
                ),
                "accepted_by_current_thresholds": bool(
                    match["ransac_inliers"] >= int(engine.config["MIN_PNP_INLIERS"])
                    and match["median_inlier_error_px"] <= PNP_THRESHOLD
                    and match["ransac_inliers"] / match["pnp_correspondences"] >= 0.1
                ),
            }
            if best is None or (
                row["inliers"],
                -row["median_inlier_reprojection_px"],
                row["similarity_score"],
            ) > (
                best["row"]["inliers"],
                -best["row"]["median_inlier_reprojection_px"],
                best["row"]["similarity_score"],
            ):
                best = {"row": row, "visual": result}
        if best:
            rows.append(best["row"])
            visual_results[name] = best["visual"]
        else:
            rows.append(
                {
                    "query_reference": name,
                    "camera_id": int(image.camera_id),
                    "camera_model": str(camera.model),
                    "camera_size": [int(camera.width), int(camera.height)],
                    "source_intrinsics": source_intrinsics.tolist(),
                    "working_intrinsics": working_intrinsics.tolist(),
                    "working_size": [int(working.shape[1]), int(working.shape[0])],
                    "query_sift_keypoints": len(keypoints),
                    "retrieval_method": method,
                    "self_reference_excluded": True,
                    "best_reference": None,
                    "retrieval_rank": None,
                    "similarity_score": None,
                    "pnp_correspondences": 0,
                    "inliers": 0,
                    "inlier_ratio": 0,
                    "median_inlier_reprojection_px": None,
                    "camera_center_error_colmap_units": None,
                    "rotation_error_degrees": None,
                    "accepted_by_current_thresholds": False,
                }
            )
    return rows, visual_results


def _save_visualizations(
    output,
    query_work,
    reference_work,
    cache,
    actual_errors,
    actual_inlier_mask,
    depth,
    ply_xyz,
    colmap_xyz,
    colmap_centers,
    camera_center,
    evaluation,
):
    height, width = query_work.shape[:2]
    fig, axes = plt.subplots(2, 1, figsize=(15, 7))
    axes[0].imshow(query_work)
    axes[0].set_title("Imported query.jpg (960 x 290 working image)")
    axes[1].imshow(reference_work)
    axes[1].set_title("Top PnP reference: 0000000348.png (road scene)")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "01_query_and_reference.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    figure, axis = plt.subplots(figsize=(18, 7))
    axis.imshow(np.concatenate([query_work, reference_work], axis=1))
    reference_points = [
        cache["reference_keypoints"][match.trainIdx].pt for match in cache["good"]
    ]
    for index, (query_point, reference_point) in enumerate(
        zip(cache["query_points"], reference_points)
    ):
        color = "#20d080" if actual_inlier_mask[index] else "#ff554f"
        axis.plot(
            [query_point[0], reference_point[0] + width],
            [query_point[1], reference_point[1]],
            color=color,
            linewidth=0.7,
            alpha=0.72,
        )
        axis.scatter(
            [query_point[0], reference_point[0] + width],
            [query_point[1], reference_point[1]],
            s=10,
            color=color,
        )
    axis.axis("off")
    axis.set_title(
        f"{len(cache['good'])} Lowe matches: "
        f"{int(actual_inlier_mask.sum())} residual inliers / "
        f"{int((~actual_inlier_mask).sum())} outliers (4 px)"
    )
    figure.tight_layout()
    figure.savefig(output / "02_query_reference_matches.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    if depth is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        image = axes[0].imshow(depth, cmap="viridis")
        fig.colorbar(image, ax=axes[0], label="stored depth value")
        axes[0].set_title("Legacy depth.npy (all values = 5.0)")
        invalid = ~np.isfinite(depth)
        axes[1].imshow(invalid, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(
            f"Non-finite pixels: {int(invalid.sum())}/{depth.size}; no validity mask"
        )
        fig.suptitle("Not used by Flask; depth provenance and scale unverified")
        fig.tight_layout()
        fig.savefig(output / "03_legacy_depth_audit.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(16, 5))
    axis.imshow(query_work)
    for index, point in enumerate(cache["query_points"]):
        axis.scatter(
            point[0],
            point[1],
            s=42,
            facecolors="none",
            edgecolors="#20d080" if actual_inlier_mask[index] else "#ff554f",
            linewidths=1.2,
        )
    axis.axis("off")
    axis.set_title(
        f"Query SIFT keypoints: 287; PnP inputs: {len(cache['good'])}; "
        "green <=4 px, red >4 px"
    )
    fig.tight_layout()
    fig.savefig(output / "04_query_keypoints_and_inliers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    colors = np.where(actual_inlier_mask, "#1dbb79", "#ed6258")
    axis.scatter(
        np.arange(1, len(actual_errors) + 1),
        actual_errors,
        c=colors,
        s=55,
        edgecolors="black",
        linewidths=0.3,
    )
    axis.axhline(PNP_THRESHOLD, color="#f6b84a", linestyle="--", label="RANSAC threshold = 4 px")
    axis.set_yscale("symlog", linthresh=1)
    axis.set_xlabel("Lowe-ratio correspondence")
    axis.set_ylabel("Reprojection error (px, symlog)")
    axis.set_title("Residuals for the pose persisted by Flask")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "05_reprojection_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(12, 9))
    axis = fig.add_subplot(111, projection="3d")
    step = max(1, len(ply_xyz) // 50000)
    axis.scatter(
        ply_xyz[::step, 0],
        ply_xyz[::step, 1],
        ply_xyz[::step, 2],
        s=0.12,
        c="#80b5c5",
        alpha=0.13,
        label="Gaussian PLY centers (stride sample)",
    )
    axis.scatter(
        colmap_xyz[:, 0],
        colmap_xyz[:, 1],
        colmap_xyz[:, 2],
        s=0.15,
        c="#f0c66e",
        alpha=0.18,
        label="COLMAP points3D",
    )
    axis.scatter(
        colmap_centers[:, 0],
        colmap_centers[:, 1],
        colmap_centers[:, 2],
        s=8,
        c="#65d9ad",
        alpha=0.8,
        label="COLMAP cameras",
    )
    axis.scatter(
        cache["world_points"][actual_inlier_mask, 0],
        cache["world_points"][actual_inlier_mask, 1],
        cache["world_points"][actual_inlier_mask, 2],
        s=18,
        c="#20d080",
        edgecolors="black",
        linewidths=0.2,
        label="Query PnP inlier 3D points",
    )
    axis.scatter(
        cache["world_points"][~actual_inlier_mask, 0],
        cache["world_points"][~actual_inlier_mask, 1],
        cache["world_points"][~actual_inlier_mask, 2],
        s=18,
        c="#ff554f",
        marker="x",
        label="Query PnP outlier 3D points",
    )
    axis.scatter(
        *camera_center,
        s=110,
        marker="*",
        c="#ff4d56",
        edgecolors="white",
        linewidths=0.5,
        label="Query pose (UNCERTAIN, 6 inliers)",
    )
    axis.set_xlabel("COLMAP X")
    axis.set_ylabel("COLMAP Y")
    axis.set_zlabel("COLMAP Z")
    axis.set_title("Point-cloud overlap and estimated query camera")
    axis.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "06_ply_colmap_camera_pose.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    labels = [row["query_reference"].replace(".png", "") for row in evaluation]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(labels, [row["query_sift_keypoints"] for row in evaluation], color="#3da8c5")
    axes[0].set_title("Known-reference SIFT points")
    axes[1].bar(labels, [row["inliers"] for row in evaluation], color="#48b58c")
    axes[1].axhline(15, color="#d77", linestyle="--", label="acceptance threshold")
    axes[1].set_title("Best cross-view PnP inliers")
    axes[1].legend(fontsize=8)
    axes[2].bar(
        labels,
        [row["camera_center_error_colmap_units"] or 0 for row in evaluation],
        color="#e0a857",
    )
    axes[2].set_title("Camera-center error (COLMAP units)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=60)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Five known road images; own COLMAP K scaled to work size; self image excluded")
    fig.tight_layout()
    fig.savefig(output / "07_reference_cross_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
    with app.app_context():
        scene = app.extensions["scene_manager"].get_scene("scene-routiere-finale-2")
        engine = app.extensions["localization_engine"]
        index = engine._load_index(scene)
        reconstruction = index["reconstruction"]
        references = index["references"]
        cv = engine._cv2()
        sift = cv.SIFT_create(nfeatures=5000)
        query_rgb, query_bgr = _read_image(QUERY_PATH, cv)
        query_work, _, _ = engine._resize(query_rgb)
        query_keypoints, query_descriptors = sift.detectAndCompute(
            cv.cvtColor(query_work, cv.COLOR_RGB2GRAY), None
        )
        selected, retrieval_method = engine._select_references(references, query_work)
        query_intrinsics = engine._query_intrinsics(
            query_bgr.shape[1],
            query_bgr.shape[0],
            query_work.shape[1],
            query_work.shape[0],
        )

        candidate_rows = []
        candidate_caches = {}
        for rank, (reference, score) in enumerate(selected, 1):
            (
                reference_work,
                reference_keypoints,
                mapped_reference_keypoints,
                _,
            ) = _features_for_reference(
                engine, reconstruction, reference, sift, cv
            )
            match, cache = _run_pnp(
                engine,
                query_keypoints,
                query_descriptors,
                reference,
                query_intrinsics,
                20261107 + rank,
            )
            row = {
                "rank": rank,
                "reference": reference["name"],
                "similarity_score": float(score),
                "score_method": retrieval_method,
                "reference_sift_keypoints": len(reference_keypoints),
                "reference_sift_mapped_to_colmap": len(reference["descriptors"]),
                "mapped_distinct_xyz": int(len(np.unique(reference["xyz"], axis=0))),
                **match,
            }
            candidate_rows.append(row)
            if cache is not None:
                candidate_caches[reference["name"]] = {
                    **cache,
                    "reference_keypoints": reference_keypoints,
                    "mapped_reference_keypoints": mapped_reference_keypoints,
                    "reference_work": reference_work,
                }

        record = _get_latest_query_record()
        actual_reference = record["metrics"]["reference_name"]
        if actual_reference not in candidate_caches:
            raise RuntimeError(f"Stored best reference missing from top 10: {actual_reference}")
        cache = candidate_caches[actual_reference]
        pose = record["pose"]
        actual_intrinsics = np.asarray(record["metrics"]["intrinsics"], dtype=np.float64)
        rvec, _ = cv.Rodrigues(np.asarray(pose["rotation_matrix"], dtype=np.float64))
        projected, _ = cv.projectPoints(
            cache["world_points"],
            rvec,
            np.asarray(pose["translation"], dtype=np.float64),
            actual_intrinsics,
            None,
        )
        actual_errors = np.linalg.norm(
            projected.reshape(-1, 2) - cache["query_points"], axis=1
        )
        actual_inlier_mask = actual_errors <= PNP_THRESHOLD

        rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
        translation = np.asarray(pose["translation"], dtype=np.float64)
        world_to_camera = np.asarray(pose["world_to_camera"], dtype=np.float64)
        camera_to_world = np.linalg.inv(world_to_camera)
        transform_checks = {
            "R_orthogonality_frobenius": float(
                np.linalg.norm(rotation.T @ rotation - np.eye(3))
            ),
            "R_determinant": float(np.linalg.det(rotation)),
            "homogeneous_inverse_roundtrip_frobenius": float(
                np.linalg.norm(world_to_camera @ camera_to_world - np.eye(4))
            ),
            "camera_center_minus_RTt": (-rotation.T @ translation).tolist(),
            "camera_center_inverse_transform": camera_to_world[:3, 3].tolist(),
            "stored_camera_center": pose["camera_center"],
            "center_consistency_norm": float(
                np.linalg.norm(
                    camera_to_world[:3, 3] - np.asarray(pose["camera_center"])
                )
            ),
        }

        colmap_xyz = np.asarray(
            [point.xyz for point in reconstruction.points3D.values()], dtype=np.float64
        )
        ply_xyz = _load_ply_xyz(PLY_PATH)
        distances = cKDTree(ply_xyz).query(colmap_xyz, k=1, workers=-1)[0]
        inside = np.all(
            (colmap_xyz >= ply_xyz.min(axis=0))
            & (colmap_xyz <= ply_xyz.max(axis=0)),
            axis=1,
        )
        overlap = {
            "ply_vertices": int(len(ply_xyz)),
            "colmap_points": int(len(colmap_xyz)),
            "ply_bounds_min": ply_xyz.min(axis=0).tolist(),
            "ply_bounds_max": ply_xyz.max(axis=0).tolist(),
            "colmap_bounds_min": colmap_xyz.min(axis=0).tolist(),
            "colmap_bounds_max": colmap_xyz.max(axis=0).tolist(),
            "colmap_inside_ply_aabb_fraction": float(inside.mean()),
            "colmap_to_ply_distance_quantiles": np.quantile(
                distances, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
            ).tolist(),
            "colmap_fraction_within_nearest_distance": {
                str(threshold): float(np.mean(distances <= threshold))
                for threshold in (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
            },
        }

        depth = np.load(DEPTH_PATH) if DEPTH_PATH.is_file() else None
        depth_summary = {
            "path": str(DEPTH_PATH),
            "exists": depth is not None,
            "used_by_active_engine": False,
            "depth_based_3d_correspondences_sent_to_pnp": 0,
            "scientifically_usable_3d_points_from_this_file": 0,
        }
        if depth is not None:
            finite = np.isfinite(depth)
            values = depth[finite]
            depth_summary.update(
                shape=list(depth.shape),
                dtype=str(depth.dtype),
                pixels=int(depth.size),
                finite_pixels=int(finite.sum()),
                invalid_nonfinite_pixels=int((~finite).sum()),
                finite_positive_over_1e_4_pixels=int(
                    (finite & (depth > 1e-4)).sum()
                ),
                unique_finite_values=int(np.unique(values).size),
                finite_quantiles=np.quantile(
                    values, [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
                ).tolist(),
                validity_mask_present=False,
                provenance_verified=False,
            )

        evaluation_rows, _ = _cross_check_references(
            engine, reconstruction, references, sift, cv
        )

    query_status = {
        "id": record["id"],
        "status": record["status"],
        "error": record.get("error"),
        "query_path": str(QUERY_PATH),
        "query_dimensions": [int(query_bgr.shape[1]), int(query_bgr.shape[0])],
        "observed_content": "Black test card with green text 'Image Requete (Test)'",
        "is_road_scene_photo": False,
        "query_sift_keypoints": len(query_keypoints),
        "retrieval_method": retrieval_method,
        "pnp_reference": actual_reference,
        "pnp_correspondences": int(len(cache["good"])),
        "stored_ransac_inliers": int(record["metrics"]["inliers"]),
        "stored_inlier_ratio": float(record["metrics"]["inlier_ratio"]),
        "residual_inliers_at_4px": int(actual_inlier_mask.sum()),
        "residual_outliers_over_4px": int((~actual_inlier_mask).sum()),
        "inlier_error_quantiles_px": (
            np.quantile(actual_errors[actual_inlier_mask], [0, 0.25, 0.5, 0.75, 0.9, 1]).tolist()
            if actual_inlier_mask.any()
            else []
        ),
        "outlier_error_quantiles_px": (
            np.quantile(actual_errors[~actual_inlier_mask], [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1]).tolist()
            if (~actual_inlier_mask).any()
            else []
        ),
        "all_error_quantiles_px": np.quantile(
            actual_errors, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1]
        ).tolist(),
        "working_size": [int(query_work.shape[1]), int(query_work.shape[0])],
        "calib_file": str(DATA / "data" / "calib.txt"),
        "calib_source_values_fx_fy_cx_cy": (
            DATA / "data" / "calib.txt"
        ).read_text(encoding="utf-8").split(),
        "K_working": actual_intrinsics.tolist(),
        "pnp_parameters": {
            "method": "cv2.SOLVEPNP_EPNP",
            "iterationsCount": PNP_ITERATIONS,
            "reprojectionError_px": PNP_THRESHOLD,
            "confidence": PNP_CONFIDENCE,
            "distCoeffs": None,
            "post_ransac_refinement": "cv2.solvePnPRefineLM",
        },
        "dino_enabled": bool(app.config["ENABLE_DINOV2"]),
        "torch_installed": bool(importlib.util.find_spec("torch")),
        "dino_scores_available": False,
        "pose_camera_center_colmap_units": pose["camera_center"],
        "pose_euler_xyz_degrees": pose["euler_xyz_degrees"],
    }
    summary = {
        "query": query_status,
        "depth": depth_summary,
        "transform_checks": transform_checks,
        "ply_colmap_overlap": overlap,
        "top10": candidate_rows,
        "known_reference_crosschecks": evaluation_rows,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for filename, rows in (
        ("query_top10.csv", candidate_rows),
        ("known_reference_crosscheck.csv", evaluation_rows),
    ):
        with (OUTPUT / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            fieldnames = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    centers = np.asarray(
        [
            -image.cam_from_world().matrix()[:3, :3].T
            @ image.cam_from_world().matrix()[:3, 3]
            for image in reconstruction.images.values()
        ]
    )
    _save_visualizations(
        OUTPUT,
        query_work,
        cache["reference_work"],
        {
            **cache,
            "reference_keypoints": cache["mapped_reference_keypoints"],
        },
        actual_errors,
        actual_inlier_mask,
        depth,
        ply_xyz,
        colmap_xyz,
        centers,
        np.asarray(pose["camera_center"]),
        evaluation_rows,
    )

    report = [
        "# Diagnostic scientifique — query.jpg",
        "",
        "Aucun seuil ni algorithme du localiseur n’a été modifié. Le diagnostic utilise les fichiers présents et conserve les sorties sous `projet/diagnostics/query_localization_20261107/`.",
        "",
        "## Constat déterminant",
        "",
        "Le `query.jpg` importé est un carton noir avec le texte vert « Image Requete (Test) », pas une photo de route. La pose obtenue ne peut donc pas être considérée comme une localisation routière.",
        "",
        "## Retrieval et correspondances actuels",
        "",
        f"- DINOv2 n’est pas actif: ENABLE_DINOV2={query_status['dino_enabled']}; PyTorch installé={query_status['torch_installed']}. Les scores ci-dessous sont ceux du repli histogramme HSV, pas des scores DINO.",
        f"- {len(references)} références indexées; query SIFT={len(query_keypoints)}; meilleure candidate PnP={actual_reference}; {query_status['pnp_correspondences']} entrées PnP, {query_status['stored_ransac_inliers']} inliers.",
        "",
        "| Rang | Référence | Score HSV | SIFT ref. | SIFT mappés à COLMAP | paires brutes | Lowe | PnP | inliers diagnostic |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidate_rows:
        report.append(
            f"| {row['rank']} | {row['reference']} | {row['similarity_score']:.6f} | "
            f"{row['reference_sift_keypoints']} | {row['reference_sift_mapped_to_colmap']} | "
            f"{row['raw_knn_pairs']} | {row['after_lowe_ratio_0_78']} | "
            f"{row['pnp_correspondences']} | {row['ransac_inliers'] if row['ransac_inliers'] is not None else 'échec'} |"
        )
    report += [
        "",
        "## Profondeur",
        "",
        f"- `depth.npy`: {depth_summary.get('pixels', 0)} pixels, {depth_summary.get('finite_pixels', 0)} finis, {depth_summary.get('invalid_nonfinite_pixels', 0)} non finis; valeurs uniques finies={depth_summary.get('unique_finite_values')}, quantiles={depth_summary.get('finite_quantiles')}.",
        "- Cette profondeur héritée n’est pas lue par le pipeline Flask actif. Elle est constante 5.0, sans masque de validité ni provenance vérifiée: aucune rétroprojection 3D n’est scientifiquement défendable. Points profondeur envoyés par le pipeline actuel: 0; celui-ci utilise des XYZ COLMAP.",
        "",
        "## PnP, calibration et résidus",
        "",
        f"- solvePnPRansac: EPNP, iterationsCount={PNP_ITERATIONS}, seuil={PNP_THRESHOLD} px, confidence={PNP_CONFIDENCE}, distCoeffs=None; raffinement `solvePnPRefineLM`.",
        f"- calib.txt: fx=fy=716.614, cx=621, cy=187.5 à 1242×375; K travail={actual_intrinsics.tolist()} à 960×290.",
        f"- Résultat persisté: {query_status['stored_ransac_inliers']}/{query_status['pnp_correspondences']} inliers, statut `{record['status']}`. Sous le seuil actuel de 15.",
        f"- Résidus avec pose persistée: quantiles inliers={query_status['inlier_error_quantiles_px']}; outliers={query_status['outlier_error_quantiles_px']} px.",
        "",
        "## Conventions de pose et repères",
        "",
        f"- Contrôles: ||RᵀR−I||F={transform_checks['R_orthogonality_frobenius']:.3g}, det(R)={transform_checks['R_determinant']:.9f}, fermeture inverse={transform_checks['homogeneous_inverse_roundtrip_frobenius']:.3g}, cohérence centre caméra={transform_checks['center_consistency_norm']:.3g} unités.",
        f"- Recouvrement: {overlap['colmap_inside_ply_aabb_fraction']:.2%} des points COLMAP dans les bornes PLY; distances nearest PLY ≤0.05: {overlap['colmap_fraction_within_nearest_distance']['0.05']:.2%}, ≤0.5: {overlap['colmap_fraction_within_nearest_distance']['0.5']:.2%}. Cela indique un fort recouvrement, pas une vérification d’échelle métrique.",
        "",
        "## Images routières connues",
        "",
        "Tests croisés sur cinq images COLMAP: la query test est retirée de l’index candidat et chaque test utilise sa calibration caméra COLMAP ajustée à 960×290.",
        "Ces cinq tests sont des contrôles de cohérence internes à la reconstruction, pas un jeu indépendant: le meilleur candidat est à une image d’écart (même séquence, vues adjacentes). Ils ne démontrent pas la généralisation à une nouvelle caméra, route ou scène.",
        "",
        "| Query connue | meilleure référence | correspondances | inliers | erreur centre unités COLMAP | erreur rotation degrés | seuils actuels |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in evaluation_rows:
        center_error = row["camera_center_error_colmap_units"]
        rotation_error = row["rotation_error_degrees"]
        report.append(
            f"| {row['query_reference']} | {row['best_reference'] or '—'} | "
            f"{row['pnp_correspondences']} | {row['inliers']} | "
            f"{center_error if center_error is not None else '—'} | "
            f"{rotation_error if rotation_error is not None else '—'} | "
            f"{'oui' if row['accepted_by_current_thresholds'] else 'non'} |"
        )
    report += [
        "",
        "## Causes probables et actions prioritaires",
        "",
        "1. Remplacer `query.jpg` par une vraie photo routière; c’est la cause immédiate dominante.",
        "2. Activer DINOv2 avec PyTorch et poids disponibles, puis comparer le rappel top-k. Les résultats actuels utilisent seulement l’histogramme HSV.",
        "3. Mesurer les pertes SIFT/Lowe sur les requêtes routières réelles; actuellement les 287 keypoints query ne donnent que 11–24 matches Lowe par candidate.",
        "4. Inspecter et rejeter les faux matches 2D–3D avant de modifier des paramètres.",
        "5. Pour le pipeline profondeur du notebook, générer de nouveau une vraie profondeur rendue avec alpha; ne pas employer le `depth.npy` constant.",
        "6. Conserver les seuils PnP jusqu’à une évaluation indépendante.",
        "",
        "## Fichiers",
        "",
        "`query_top10.csv`, `known_reference_crosscheck.csv`, `diagnostic_summary.json` et les figures `01` à `07` du dossier.",
        "",
        "Aucune partie du moteur de localisation n’a été changée.",
    ]
    (OUTPUT / "DIAGNOSTIC_FR.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
