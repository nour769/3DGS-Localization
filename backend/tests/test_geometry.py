import unittest

import numpy as np

from app.localization.geometry import (
    camera_center_world,
    quaternion_to_euler_xyz_degrees,
    rotation_to_quaternion_wxyz,
    scaled_intrinsics,
)
from app.localization.engine import LocalizationEngine, LocalizationError


class GeometryTests(unittest.TestCase):
    def test_intrinsics_scale_independently_per_axis(self):
        matrix = scaled_intrinsics(716.614, 700.0, 621.0, 187.0, 1242, 375, 621, 750)
        np.testing.assert_allclose(
            matrix,
            [[358.307, 0, 310.5], [0, 1400.0, 374.0], [0, 0, 1]],
            rtol=1e-8,
        )

    def test_camera_center_uses_world_to_camera_convention(self):
        rotation = np.eye(3)
        translation = np.array([4.0, -2.0, 8.0])
        np.testing.assert_allclose(camera_center_world(rotation, translation), -translation)

    def test_rotation_quaternion_and_euler_round_trip_for_identity(self):
        quaternion = rotation_to_quaternion_wxyz(np.eye(3))
        np.testing.assert_allclose(quaternion, [1, 0, 0, 0])
        np.testing.assert_allclose(quaternion_to_euler_xyz_degrees(quaternion), [0, 0, 0])

    def test_query_calibration_file_is_loaded_and_scaled_to_working_image(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            calibration_path = Path(temporary) / "calib.txt"
            calibration_path.write_text(
                "716.614 716.614 621.0 187.5", encoding="utf-8"
            )
            engine = LocalizationEngine(
                {
                    "QUERY_CALIB_PATH": str(calibration_path),
                    "QUERY_WIDTH": 1242,
                    "QUERY_HEIGHT": 375,
                    "QUERY_FX": 1,
                    "QUERY_FY": 1,
                    "QUERY_CX": 1,
                    "QUERY_CY": 1,
                }
            )
            matrix = engine._query_intrinsics(1242, 375, 960, 290)
        np.testing.assert_allclose(
            matrix,
            [
                [716.614 * 960 / 1242, 0, 480],
                [0, 716.614 * 290 / 375, 145],
                [0, 0, 1],
            ],
            atol=1e-8,
        )

    def test_invalid_query_calibration_file_is_an_error(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            calibration_path = Path(temporary) / "calib.txt"
            calibration_path.write_text("716.614 716.614 621", encoding="utf-8")
            engine = LocalizationEngine(
                {
                    "QUERY_CALIB_PATH": str(calibration_path),
                    "QUERY_WIDTH": 1242,
                    "QUERY_HEIGHT": 375,
                    "QUERY_FX": 716.614,
                    "QUERY_FY": 716.614,
                    "QUERY_CX": 621,
                    "QUERY_CY": 187.5,
                }
            )
            with self.assertRaisesRegex(LocalizationError, "exactly four"):
                engine._query_intrinsics(1242, 375, 960, 290)


if __name__ == "__main__":
    unittest.main()
