from __future__ import annotations

from dataclasses import dataclass, field, replace

import mujoco
import numpy as np
from experiments.daisy_cable_replanning.cable_demo import DemoScript
from experiments.daisy_cable_replanning.collisions import ArmCollisionPairs
from semantic_digital_twin.adapters.multi_sim import MujocoSim, MujocoSynchronizer
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.collision_checking.collision_detector import ClosestPoints
from semantic_digital_twin.collision_checking.collision_matrix import CollisionMatrix
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from typing_extensions import Self

# %% viewpoint


@dataclass(frozen=True)
class OrbitViewpoint:
    """
    Where a camera orbiting a point looks at the scene from.
    """

    look_at: tuple[float, float, float]
    """
    The point the camera looks at and orbits, in the world frame.
    """

    distance: float
    """
    Distance of the camera from the point, in metres.
    """

    azimuth: float = 135.0
    """
    Rotation of the camera about the vertical through the point, in degrees.
    """

    elevation: float = -25.0
    """
    Angle of the camera's line of sight above the horizontal, in degrees.
    """

    def orbited(self, azimuth_change: float, elevation_change: float) -> Self:
        """
        :return: This viewpoint moved around the point by the angles, in degrees,
            looking from at most straight above or below.
        """
        return replace(
            self,
            azimuth=self.azimuth + azimuth_change,
            elevation=float(np.clip(self.elevation + elevation_change, -89.0, 89.0)),
        )

    def focused_on(self, point: np.ndarray, distance: float) -> Self:
        """
        :return: This viewpoint looking at the point from the distance, in metres.
        """
        return replace(
            self,
            look_at=(float(point[0]), float(point[1]), float(point[2])),
            distance=distance,
        )

    def zoomed(self, factor: float) -> Self:
        """
        :return: This viewpoint with its distance multiplied by the factor.
        """
        return replace(self, distance=self.distance * factor)

    def camera(self) -> mujoco.MjvCamera:
        """
        :return: A free MuJoCo camera at this viewpoint.
        """
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = self.look_at
        camera.distance = self.distance
        camera.azimuth = self.azimuth
        camera.elevation = self.elevation
        return camera


# %% robot snapshot


@dataclass
class RobotSnapshot:
    """
    DAiSy next to the nominal cable workcell of the demo, put into joint states and
    rendered.

    .. note:: Rendering next to a Qt window needs MuJoCo's EGL backend, which MuJoCo
        chooses once, when it is imported: start the process with ``MUJOCO_GL=egl``.
    """

    world: World
    """
    The world of DAiSy and the workcell.
    """

    robot: DAiSy
    """
    The robot shown.
    """

    connections: list[ActiveConnection1DOF]
    """
    The joints a shown joint state gives positions of, in order.
    """

    arm_pairs: ArmCollisionPairs
    """
    The bodies of the arms and the pairs of them that may not touch.
    """

    largest_image_width: int = 640
    """
    Width of the largest image MuJoCo renders, that of its default offscreen
    framebuffer, in pixels.
    """

    largest_image_height: int = 480
    """
    Height of the largest image MuJoCo renders, in pixels.
    """

    _simulation: MujocoSim = field(kw_only=True, repr=False)
    """
    The MuJoCo mirror of the world the images are rendered from.
    """

    _self_collision_matrix: CollisionMatrix = field(init=False, repr=False)
    """
    The pairs of robot bodies involving an arm, checked for contact.
    """

    def __post_init__(self):
        self._self_collision_matrix = self.arm_pairs.self_collision_matrix(distance=0.0)

    @classmethod
    def of_cable_workcell(cls, joint_names: list[str]) -> Self:
        """
        :param joint_names: The joints a shown joint state gives positions of.
        :return: A snapshot of DAiSy with parked arms next to the nominal workcell.
        """
        world = URDFParser.from_file(file_path=DAiSy.get_ros_file_path()).parse()
        DAiSy.from_world(world)
        robot = world.get_semantic_annotations_by_type(DAiSy)[0]
        for arm in robot.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        # Before the workcell is added, so the post and hanger are not robot bodies.
        arm_pairs = ArmCollisionPairs(world=world, robot=robot)
        demo = DemoScript.daisy_cable_demo().load()
        demo.CableWorkcell.find_or_add_to(world, robot, demo.CableWorkcellLayout())
        return cls(
            world=world,
            robot=robot,
            connections=[world.get_connection_by_name(name) for name in joint_names],
            arm_pairs=arm_pairs,
            _simulation=cls._paused_simulation(world),
        )

    @staticmethod
    def _paused_simulation(world: World) -> MujocoSim:
        """
        :return: A headless MuJoCo mirror of the world, paused so that stepping it only
            brings the body poses up to date with the joint positions.
        """
        simulation = MujocoSim(world=world, headless=True)
        simulation.synchronizer.sync_rate_hz = (
            MujocoSynchronizer.UNTHROTTLED_SYNC_RATE_HZ
        )
        simulation.simulator.start(simulate_in_thread=False, render_in_thread=False)
        simulation.simulator.pause()
        return simulation

    def overview(self) -> OrbitViewpoint:
        """
        :return: A viewpoint that frames the whole world, looking at the height the
            arms work at rather than at the middle of the table legs.
        """
        bounds = np.asarray(RayTracer(self.world).scene.bounds)
        center = bounds.mean(axis=0)
        working_height = bounds[0, 2] + 0.75 * (bounds[1, 2] - bounds[0, 2])
        return OrbitViewpoint(
            look_at=(float(center[0]), float(center[1]), float(working_height)),
            distance=float(np.linalg.norm(bounds[1] - bounds[0])) * 0.9,
        )

    def show(self, positions: list[float]) -> None:
        """
        Put the joints into the positions.

        :param positions: Positions of :attr:`connections`.
        """
        with self.world.batch_state_changes():
            for connection, position in zip(self.connections, positions):
                connection.position = position

    def render(self, viewpoint: OrbitViewpoint, width: int, height: int) -> np.ndarray:
        """
        :param viewpoint: Where the camera looks from.
        :param width: Width of the image, at most :attr:`largest_image_width`.
        :param height: Height of the image, at most :attr:`largest_image_height`.
        :return: An RGB image of the world in its current joint state.
        """
        simulator = self._simulation.simulator
        simulator.step()
        return simulator.capture_rgb(
            height=height, width=width, free_camera=viewpoint.camera()
        ).result

    def contacts(self) -> list[ClosestPoints]:
        """
        :return: The pairs of robot bodies involving an arm that touch in the current
            joint state, the deepest first.
        """
        detector = self.world.collision_manager.collision_detector
        return sorted(
            (
                contact
                for contact in detector.check_collisions(
                    self._self_collision_matrix
                ).contacts
                if contact.distance <= 0.0
            ),
            key=lambda contact: contact.distance,
        )

    def stop(self) -> None:
        """
        Stop the MuJoCo mirror.
        """
        self._simulation.simulator.stop()
