from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import pi
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy
from physics_simulators.base_simulator import SimulatorState

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Color, Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_modification import (
    AddConnectionModification,
)

if TYPE_CHECKING:
    from semantic_digital_twin.adapters.multi_sim import MujocoEquality, MujocoSim

logger = logging.getLogger(__name__)


@dataclass
class CableConfig:
    """
    Configuration for building a cable as a kinematic chain of rigid bodies.
    """

    segment_count: int = 10
    segment_length: float = 0.03
    radius: float = 0.005
    mass_per_segment: float = 0.005
    color: Color = field(default_factory=lambda: Color(1, 0, 0, 1))


@dataclass
class Cable:
    """
    A cable assembled from rigid bodies linked by connect equality constraints.

    Each segment is a :class:`Body` with a :class:`Cylinder` shape and a free
    joint (child of ``world``).  Consecutive segments are linked by MuJoCo
    ``connect`` equality constraints which act as ball-and-socket joints: they
    keep the segment endpoints together while allowing free rotation.

    The start end can be attached to an arbitrary body (e.g. a robot gripper)
    and the end can optionally be fixed to another body (e.g. a table or wall)
    through additional connect constraints.
    """

    segments: List[Body] = field(default_factory=list)
    connections: List[Connection6DoF] = field(default_factory=list)
    constraints: List[Any] = field(default_factory=list)
    config: CableConfig = field(default_factory=CableConfig)


def build_cable(
    config: CableConfig,
    world: World,
    parent_body: Optional[Body] = None,
) -> Cable:
    """
    Build a cable in the world for MuJoCo simulation.

    MuJoCo requires ``mjJNT_FREE`` bodies to be direct children of the world
    body.  This function therefore places every segment as a top-level free-
    joint body and links consecutive segments with ``mjEQ_CONNECT`` equality
    constraints (ball-and-socket joints at the segment endpoints).

    :param config: Configuration describing segment count, geometry, and
        physical properties.
    :param world: The world to which the cable segments and constraints
        are added.
    :param parent_body: Optional body to which the first cable segment
        will be attached via a connect equality constraint.  When provided,
        the cable segments are positioned near this body's world location.
    :return: A :class:`Cable` holding references to every segment body,
        connection, and constraint.
    """
    segments: List[Body] = []
    connections: List[Connection6DoF] = []
    constraints: List = []
    half = config.segment_length / 2.0

    import mujoco
    from semantic_digital_twin.adapters.multi_sim import MujocoEquality

    # Rotate cylinder from its native z-axis to the cable direction (x-axis).
    # A 90 deg pitch about y maps z -> x.
    cylinder_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0.0, 0.0, 0.0, 0.0, pi / 2.0, 0.0
    )

    # Capture root BEFORE entering modify_world to avoid the FK root-detection
    # assertion that fires when many bodies still lack parent connections.
    if world.root is None:
        _root = Body(name=PrefixedName("world"))
        with world.modify_world():
            world.add_kinematic_structure_entity(_root)
        root = _root
    else:
        root = world.root

    if parent_body is not None and parent_body._world is None:
        with world.modify_world():
            world.add_kinematic_structure_entity(parent_body)

    # Compute the initial anchor position for the cable.
    # When a parent_body is given, the first segment connects to the parent
    # at the parent's world origin; the segment's -half endpoint is pulled to
    # that point, so the segment centre sits at parent_pos + half along x.
    if parent_body is not None:
        try:
            parent_pose = parent_body.global_transform.evaluate()
            base_x = float(parent_pose[0, 3]) + half
            base_y = float(parent_pose[1, 3])
            base_z = float(parent_pose[2, 3])
        except Exception:
            logger.warning(
                "Could not compute parent body pose; placing cable at origin"
            )
            base_x = half
            base_y = 0.0
            base_z = 0.0
    else:
        base_x = 0.0
        base_y = 0.0
        base_z = 0.0

    with world.modify_world():
        for i in range(config.segment_count):
            cyl = Cylinder(
                width=2.0 * config.radius,
                height=config.segment_length,
                color=config.color,
                origin=cylinder_origin,
            )
            body = Body(
                name=PrefixedName(f"cable_segment_{i}"),
                visual=ShapeCollection([cyl]),
                collision=ShapeCollection([cyl]),
            )
            world.add_kinematic_structure_entity(body)
            segments.append(body)

        for i, segment in enumerate(segments):
            connection = Connection6DoF.create_with_dofs(
                world=world,
                parent=root,
                child=segment,
                name=PrefixedName(f"cable_joint_{i}"),
            )
            world.add_connection(connection)
            connections.append(connection)

            offset = i * config.segment_length
            world.state[connection.x.id].position = base_x + offset
            world.state[connection.y.id].position = base_y
            world.state[connection.z.id].position = base_z
            world.state[connection.qw.id].position = 1.0
            world.state[connection.qx.id].position = 0.0
            world.state[connection.qy.id].position = 0.0
            world.state[connection.qz.id].position = 0.0

    # Connect equalities are added in a separate modify_world context to avoid
    # FK confusion during the first context exit.
    with world.modify_world():
        for i in range(config.segment_count - 1):
            constraint = MujocoEquality(
                type=mujoco.mjtEq.mjEQ_CONNECT,
                object_type=mujoco.mjtObj.mjOBJ_BODY,
                name_1=f"cable_segment_{i}",
                name_2=f"cable_segment_{i + 1}",
                data=[half, 0.0, 0.0, -half, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            )
            world.simulator_additional_properties.append(constraint)
            constraints.append(constraint)

        if parent_body is not None:
            constraint = MujocoEquality(
                type=mujoco.mjtEq.mjEQ_CONNECT,
                object_type=mujoco.mjtObj.mjOBJ_BODY,
                name_1=parent_body.name.name,
                name_2="cable_segment_0",
                data=[0.0, 0.0, 0.0, -half, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            )
            world.simulator_additional_properties.append(constraint)
            constraints.append(constraint)

    return Cable(
        segments=segments,
        connections=connections,
        constraints=constraints,
        config=config,
    )


def attach_cable_end(
    cable: Cable,
    world: World,
    target_body: Body,
    end: str = "start",
) -> None:
    """
    Attach one end of a cable to a target body with a connect equality
    constraint.

    :param cable: The cable whose end will be attached.
    :param world: The world containing the cable and target body.
    :param target_body: The body to attach the cable end to.
    :param end: ``"start"`` to attach the first segment or ``"end"`` to
        attach the last segment.
    """
    import mujoco
    from semantic_digital_twin.adapters.multi_sim import MujocoEquality

    if end == "start":
        segment_name = cable.segments[0].name.name
        point_on_segment = [-cable.config.segment_length / 2.0, 0.0, 0.0]
    elif end == "end":
        segment_name = cable.segments[-1].name.name
        point_on_segment = [cable.config.segment_length / 2.0, 0.0, 0.0]
    else:
        raise ValueError(f"end must be 'start' or 'end', got {end!r}")

    with world.modify_world():
        constraint = MujocoEquality(
            type=mujoco.mjtEq.mjEQ_CONNECT,
            object_type=mujoco.mjtObj.mjOBJ_BODY,
            name_1=target_body.name.name,
            name_2=segment_name,
            data=[0.0, 0.0, 0.0, *point_on_segment, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        world.simulator_additional_properties.append(constraint)
        cable.constraints.append(constraint)


@dataclass
class CableSimulation:
    """
    Background MuJoCo simulation of a cable in a world.

    Builds a cable via :func:`build_cable`, creates a headless
    :class:`~semantic_digital_twin.adapters.multi_sim.MujocoSim`, and
    provides methods to start/stop the background physics thread and to
    grasp or release cable segments from robot grippers.

    Listens for world model changes so that when a
    :class:`~semantic_digital_twin.world_description.connections.FixedConnection`
    is added to a cable segment (e.g. by a
    :class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction`),
    the corresponding MuJoCo body is automatically attached to its new
    parent body in the running physics simulation.

    Usage::

        cable_sim = CableSimulation(config, world, parent_body=hanger)
        cable_sim.start()
        # ... robot plan with PickUpAction(cable_segment, ...) ...
        # cable segment is automatically grasped in MuJoCo via attach()
        cable_sim.stop()
    """

    config: CableConfig
    """
    Cable geometry and physical properties.
    """

    world: World
    """
    The semantic digital twin world containing the robot and cable.
    """

    parent_body: Optional[Body] = None
    """
    Optional body to which the cable end is initially attached
    (e.g. a cable hanger).
    """

    sync_rate_hz: float = 30.0
    """
    How often sim→world position sync occurs (Hz). Set ≤0 to disable.
    """

    cable: Cable = field(init=False)
    """
    The built cable holding segment bodies, connections, and constraints.
    """

    multi_sim: MujocoSim = field(init=False)
    """
    The underlying :class:`MujocoSim` running the background physics.
    """

    _started: bool = field(init=False, default=False)
    _is_paused: bool = field(init=False, default=False)

    def __post_init__(self):
        from semantic_digital_twin.adapters.multi_sim import MujocoSim

        self.cable = build_cable(
            config=self.config,
            world=self.world,
            parent_body=self.parent_body,
        )
        self.multi_sim = MujocoSim(world=self.world, headless=True)
        self.multi_sim.synchronizer.sync_rate_hz = self.sync_rate_hz
        self._segment_ids = {s.id for s in self.cable.segments}
        self._register_model_callback()

    def _register_model_callback(self) -> None:
        self.world._model_manager.model_change_callbacks.append(self)

    def _unregister_model_callback(self) -> None:
        try:
            self.world._model_manager.model_change_callbacks.remove(self)
        except ValueError:
            pass

    def notify_model_change(self, **kwargs) -> None:
        if not self._is_paused:
            self.on_model_change(**kwargs)

    def on_model_change(self, **kwargs) -> None:
        """
        Detect when a cable segment is reparented (e.g. by PickUpAction)
        and automatically attach or detach the corresponding MuJoCo body.
        """
        if not self._started:
            return
        try:
            modifications = self.world._model_manager.model_modification_blocks[-1]
        except (IndexError, AttributeError):
            return

        for modification in modifications:
            if isinstance(modification, AddConnectionModification):
                connection = modification.connection
                if not isinstance(connection, FixedConnection):
                    continue
                child = connection.child
                if child.id not in self._segment_ids:
                    continue
                segment_index = self.cable.segments.index(child)
                parent_body_name = connection.parent.name.name
                self.grasp(
                    gripper_body_name=parent_body_name,
                    segment_index=segment_index,
                )

    def start(self) -> None:
        """Start the background physics simulation thread."""
        if self._started:
            return
        self.multi_sim.start_simulation()
        self._started = True
        logger.info("Cable simulation started")

    def stop(self) -> None:
        """Stop the background physics simulation thread."""
        self._unregister_model_callback()
        if not self._started:
            return
        if self.multi_sim.simulator.state != SimulatorState.STOPPED:
            self.multi_sim.stop_simulation()
        self._started = False
        logger.info("Cable simulation stopped")

    def grasp(self, gripper_body_name: str, segment_index: int = 0) -> None:
        """
        Attach a cable segment body to a robot gripper body in the
        running simulation.

        :param gripper_body_name: MuJoCo body name of the parent to
            hold the cable segment.
        :param segment_index: Index of the cable segment to grasp
            (0 = free end).
        """
        if not self._started:
            raise RuntimeError("Simulation is not running")
        if segment_index < 0 or segment_index >= len(self.cable.segments):
            raise ValueError(
                f"segment_index {segment_index} out of range "
                f" [0, {len(self.cable.segments)})"
            )
        segment_name = f"cable_segment_{segment_index}"
        self.multi_sim.simulator.callbacks["attach"](
            body_1_name=segment_name,
            body_2_name=gripper_body_name,
        )
        logger.info(
            "Cable segment %d attached to %s", segment_index, gripper_body_name
        )

    def release(self, segment_index: int = 0) -> None:
        """
        Detach a cable segment body from its parent in the running
        simulation, adding a free joint so the segment is again governed
        by physics (gravity, constraints).

        :param segment_index: Index of the cable segment to release.
        """
        if not self._started:
            raise RuntimeError("Simulation is not running")
        if segment_index < 0 or segment_index >= len(self.cable.segments):
            raise ValueError(
                f"segment_index {segment_index} out of range "
                f" [0, {len(self.cable.segments)})"
            )
        segment_name = f"cable_segment_{segment_index}"
        self.multi_sim.simulator.callbacks["detach"](
            body_name=segment_name,
            add_freejoint=True,
        )
        logger.info("Cable segment %d released", segment_index)

    def get_segment_positions(self) -> Dict[str, numpy.ndarray]:
        """
        Return the current world-frame position of every cable segment
        from the running simulation.

        :return: Mapping ``{segment_name: numpy.ndarray([x, y, z]), ...}``.
        """
        if not self._started:
            raise RuntimeError("Simulation is not running")
        segment_names = [s.name.name for s in self.cable.segments]
        result = self.multi_sim.simulator.callbacks["get_bodies_positions"](
            segment_names
        )
        return result.result
