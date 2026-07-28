from __future__ import annotations

from math import pi

import numpy as np
import pytest

from coraplex.robot_plans.actions.core.cable_grasp import (
    _gripper_orientation_from_z_axis,
)
from semantic_digital_twin.spatial_types.spatial_types import Quaternion


def _apply_rotation(quat: Quaternion, vector: np.ndarray) -> np.ndarray:
    vector4 = np.zeros(4)
    vector4[:3] = vector[:3]
    rotation_matrix = quat.to_rotation_matrix().to_np()
    return (rotation_matrix @ vector4)[:3]


class TestGripperOrientationFromZAxis:
    def test_z_axis_points_correctly(self) -> None:
        result = _gripper_orientation_from_z_axis(
            gripper_z_axis=np.array([1.0, 0.0, 0.0]),
            fallback_direction=np.array([0.0, 1.0, 0.0]),
        )

        z_result = _apply_rotation(result, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(z_result, np.array([1.0, 0.0, 0.0]), atol=1e-10)

    def test_x_axis_perpendicular_to_z_and_world_up(self) -> None:
        result = _gripper_orientation_from_z_axis(
            gripper_z_axis=np.array([1.0, 0.0, 0.0]),
            fallback_direction=np.array([0.0, 1.0, 0.0]),
        )

        x_result = _apply_rotation(result, np.array([1.0, 0.0, 0.0]))
        assert abs(np.dot(x_result, np.array([1.0, 0.0, 0.0]))) < 1e-10
        assert abs(np.dot(x_result, np.array([0.0, 0.0, 1.0]))) < 1e-10

    def test_z_parallel_to_world_up_falls_back(self) -> None:
        result = _gripper_orientation_from_z_axis(
            gripper_z_axis=np.array([0.0, 0.0, 1.0]),
            fallback_direction=np.array([1.0, 0.0, 0.0]),
        )

        z_result = _apply_rotation(result, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(z_result, np.array([0.0, 0.0, 1.0]), atol=1e-10)

    def test_degenerate_case_uses_arbitrary_horizontal(self) -> None:
        result = _gripper_orientation_from_z_axis(
            gripper_z_axis=np.array([0.0, 0.0, 1.0]),
            fallback_direction=np.array([0.0, 0.0, 1.0]),
        )

        z_result = _apply_rotation(result, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(z_result, np.array([0.0, 0.0, 1.0]), atol=1e-10)

        x_result = _apply_rotation(result, np.array([1.0, 0.0, 0.0]))
        assert abs(np.dot(x_result, np.array([0.0, 0.0, 1.0]))) < 1e-10

    @pytest.mark.parametrize("z_rotation", [0.0, pi / 2, pi])
    def test_z_rotation_preserves_z_axis(self, z_rotation: float) -> None:
        result = _gripper_orientation_from_z_axis(
            gripper_z_axis=np.array([1.0, 0.0, 0.0]),
            fallback_direction=np.array([0.0, 1.0, 0.0]),
            z_rotation=z_rotation,
        )

        z_result = _apply_rotation(result, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(z_result, np.array([1.0, 0.0, 0.0]), atol=1e-10)

    def test_z_rotation_pi_flips_x_and_y(self) -> None:
        base = _gripper_orientation_from_z_axis(
            gripper_z_axis=np.array([1.0, 0.0, 0.0]),
            fallback_direction=np.array([0.0, 1.0, 0.0]),
            z_rotation=0.0,
        )
        rotated = _gripper_orientation_from_z_axis(
            gripper_z_axis=np.array([1.0, 0.0, 0.0]),
            fallback_direction=np.array([0.0, 1.0, 0.0]),
            z_rotation=pi,
        )

        x_base = _apply_rotation(base, np.array([1.0, 0.0, 0.0]))
        x_rotated = _apply_rotation(rotated, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(x_rotated, -x_base, atol=1e-10)

        y_base = _apply_rotation(base, np.array([0.0, 1.0, 0.0]))
        y_rotated = _apply_rotation(rotated, np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(y_rotated, -y_base, atol=1e-10)

    def test_normalises_input_vectors(self) -> None:
        result = _gripper_orientation_from_z_axis(
            gripper_z_axis=np.array([2.0, 0.0, 0.0]),
            fallback_direction=np.array([0.0, 3.0, 0.0]),
        )

        z_result = _apply_rotation(result, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(z_result, np.array([1.0, 0.0, 0.0]), atol=1e-10)

    def test_result_is_orthonormal(self) -> None:
        result = _gripper_orientation_from_z_axis(
            gripper_z_axis=np.array([0.6, 0.8, 0.0]),
            fallback_direction=np.array([1.0, 0.0, 0.0]),
        )

        x_axis = _apply_rotation(result, np.array([1.0, 0.0, 0.0]))
        y_axis = _apply_rotation(result, np.array([0.0, 1.0, 0.0]))
        z_axis = _apply_rotation(result, np.array([0.0, 0.0, 1.0]))

        for axis in (x_axis, y_axis, z_axis):
            np.testing.assert_allclose(np.linalg.norm(axis), 1.0, atol=1e-10)

        for a, b in [(x_axis, y_axis), (y_axis, z_axis), (z_axis, x_axis)]:
            assert abs(np.dot(a, b)) < 1e-10
