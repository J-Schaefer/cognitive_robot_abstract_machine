from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from math import pi
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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
    RemoveConnectionModification,
)

if TYPE_CHECKING:
    from semantic_digital_twin.adapters.multi_sim import MujocoEquality, MujocoSim

logger = logging.getLogger(__name__)


class CableSimulationStrategy(enum.Enum):
    """
    Strategy for how cable segments are attached to grippers during grasping.

    .. attribute:: KINEMATIC_ATTACH

       Legacy behaviour: the cable segment body is reparented in the MuJoCo
       kinematic tree via :func:`~physics_simulators.mujoco_simulator.MujocoSimulator.attach`.
       This recompiles the entire model, discarding physics state for all
       bodies.

    .. attribute:: POSITION_OVERRIDE

       The segment body stays as a free-joint child of the world body.  A
       pre-step hook overrides the segment's ``qpos`` to follow the gripper.
       The inter-segment ``mjEQ_CONNECT`` equality constraints remain intact.
    """

    KINEMATIC_ATTACH = "kinematic_attach"
    POSITION_OVERRIDE = "position_override"


@dataclass
class CableConfig:
    r"""
    Configuration for building a cable as a kinematic chain of rigid bodies
    or as a MuJoCo native composite cable.
    """

    segment_count: int = 10
    """
    Number of rigid-body segments in the cable.
    """

    segment_length: float = 0.03
    """
    Length of each segment along its local x-axis (metres).
    """

    radius: float = 0.005
    """
    Radius of each segment's cylinder geometry (metres).
    """

    mass_per_segment: float = 0.005
    """
    Mass assigned to each segment body (kg).  Only used for
    :attr:`strategy` ``KINEMATIC_ATTACH`` and ``POSITION_OVERRIDE``;
    composite cables configure mass through MuJoCo properties.
    """

    color: Color = field(default_factory=lambda: Color(1, 0, 0, 1))
    """
    RGBA colour of the cable segments.
    """

    strategy: CableSimulationStrategy = CableSimulationStrategy.KINEMATIC_ATTACH
    """
    Which strategy to use when a cable segment is grasped.  Can be
    overridden per :class:`CableSimulation` instance via
    :attr:`CableSimulation.strategy_override`.
    """

    use_composite: bool = False
    """
    When ``True`` the cable is built using MuJoCo's native composite
    cable system instead of individual free-joint bodies linked by
    ``connect`` equality constraints.

    .. note::
       Composite cables always use :attr:`strategy`
       ``POSITION_OVERRIDE`` when a segment is grasped.
    """


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
    # A 90 deg pitch about y maps z -> x. Each segment gets its own copy
    # because the visualisation pipeline may mutate the origin's
    # reference_frame per-body.

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
            cyl_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
                0.0, 0.0, 0.0, 0.0, pi / 2.0, 0.0
            )
            cyl = Cylinder(
                width=2.0 * config.radius,
                height=config.segment_length,
                color=config.color,
                origin=cyl_origin,
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

    Builds a cable via :func:`build_cable` (or MuJoCo's native composite
    system when :attr:`CableConfig.use_composite` is ``True``), creates a
    headless :class:`~semantic_digital_twin.adapters.multi_sim.MujocoSim`,
    and provides methods to start/stop the background physics thread and to
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
        # cable segment is automatically grasped in MuJoCo
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

    strategy_override: Optional[CableSimulationStrategy] = None
    """
    When set, overrides :attr:`CableConfig.strategy` for this simulation
    instance.  When ``None`` (default) the config-level strategy is used.
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
    _segment_ids: set = field(init=False, repr=False)
    _grasped_segments: Dict[int, Tuple[str, numpy.ndarray, numpy.ndarray]] = field(
        init=False, repr=False, default_factory=dict
    )
    """
    Maps ``segment_index`` →
    ``(gripper_body_name, relative_position, relative_quaternion)``
    for segments grasped via the POSITION_OVERRIDE strategy.
    """

    _original_step_callback: Any = field(init=False, repr=False, default=None)
    """
    Saved original :meth:`MujocoSimulator.step_callback` so the position-
    override hook can be removed on release.
    """

    _composite_body_names: Dict[int, str] = field(init=False, repr=False, default_factory=dict)
    """
    Maps ``segment_index`` → MuJoCo body name for composite cables.
    """

    _position_override_installed: bool = field(init=False, repr=False, default=False)

    @property
    def _effective_strategy(self) -> CableSimulationStrategy:
        if self.strategy_override is not None:
            return self.strategy_override
        return self.config.strategy

    def __post_init__(self):
        from semantic_digital_twin.adapters.multi_sim import (
            MujocoBuilder,
            MujocoSim,
            MujocoSynchronizer,
        )

        if self.config.use_composite:
            self.cable = self._build_world_model_cable_for_composite()
        else:
            self.cable = build_cable(
                config=self.config,
                world=self.world,
                parent_body=self.parent_body,
            )
        old_builder_skip = MujocoBuilder._skip_hardware_interface_connections
        old_sync_skip = MujocoSynchronizer._skip_hardware_interface_connections
        MujocoBuilder._skip_hardware_interface_connections = True
        MujocoSynchronizer._skip_hardware_interface_connections = True
        try:
            self.multi_sim = MujocoSim(world=self.world, headless=True)
        finally:
            MujocoBuilder._skip_hardware_interface_connections = old_builder_skip
            MujocoSynchronizer._skip_hardware_interface_connections = old_sync_skip
        self.multi_sim.synchronizer.sync_rate_hz = self.sync_rate_hz

        if self.config.use_composite:
            self._build_composite_cable_in_spec()
        self._segment_ids = {s.id for s in self.cable.segments}
        self._register_model_callback()

    def _build_world_model_cable_for_composite(self) -> Cable:
        r"""
        Create minimal world-model bodies with free joints so
        :class:`PickUpAction` can target and reparent them.  The actual
        physics geometry is handled by :meth:`_build_composite_cable_in_spec`.
        """
        segments: List[Body] = []
        connections: List[Connection6DoF] = []
        world = self.world
        config = self.config

        if world.root is None:
            _root = Body(name=PrefixedName("world"))
            with world.modify_world():
                world.add_kinematic_structure_entity(_root)
            root = _root
        else:
            root = world.root

        if self.parent_body is not None and self.parent_body._world is None:
            with world.modify_world():
                world.add_kinematic_structure_entity(self.parent_body)

        half = config.segment_length / 2.0
        if self.parent_body is not None:
            try:
                parent_pose = self.parent_body.global_transform.evaluate()
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
                body = Body(name=PrefixedName(f"cable_segment_{i}"))
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
                world.state[connection.x.id].position = base_x + float(offset)
                world.state[connection.y.id].position = base_y
                world.state[connection.z.id].position = base_z
                world.state[connection.qw.id].position = 1.0
                world.state[connection.qx.id].position = 0.0
                world.state[connection.qy.id].position = 0.0
                world.state[connection.qz.id].position = 0.0

        return Cable(
            segments=segments,
            connections=connections,
            constraints=[],
            config=config,
        )

    def _build_composite_cable_in_spec(self):
        r"""
        Add a MuJoCo native composite cable to the spec and recompile.

        The bodies created by :meth:`_build_world_model_cable_for_composite`
        exist in the MuJoCo spec as empty free-joint bodies (no geoms).
        The composite bodies -- named ``cable_segment_B{idx}`` -- handle
        the actual physics.

        .. note::
           Requires MuJoCo ≥ 3.2 for the composite API.  Falls back to
           the non-composite rigid-body chain if unavailable.
        """
        import mujoco

        if not hasattr(mujoco.MjSpec, "add_composite"):
            logger.warning(
                "MuJoCo %s does not support composite objects. "
                "Falling back to non-composite cable.",
                mujoco.__version__,
            )
            return

        spec = self.multi_sim.simulator._mj_spec
        segment_count = self.config.segment_count
        segment_length = self.config.segment_length
        radius = self.config.radius
        half = segment_length / 2.0

        composite = spec.add_composite()
        composite.type = mujoco.mjtCompType.mjCOMPTYPE_CABLE
        composite.prefix = "cable_segment"
        composite.count = [segment_count, 1, 1]
        composite.spacing = [segment_length, 0, 0]
        composite.radius = [radius]

        if self.parent_body is not None:
            try:
                parent_pose = self.parent_body.global_transform.evaluate()
                base_pos = [
                    float(parent_pose[0, 3]) + half,
                    float(parent_pose[1, 3]),
                    float(parent_pose[2, 3]),
                ]
                composite.offset = base_pos
            except Exception:
                composite.offset = [half, 0.0, 0.0]
        else:
            composite.offset = [0.0, 0.0, 0.0]

        self._composite_body_names = {
            i: f"cable_segment_B{i}" for i in range(segment_count)
        }

        if spec._model is not None:
            spec.recompile(spec._model, spec._data)

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
        r"""
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
            elif isinstance(modification, RemoveConnectionModification):
                segment_index = self._find_segment_index_for_removed_connection(
                    modification
                )
                if segment_index is not None:
                    self.release(segment_index=segment_index)

    def _find_segment_index_for_removed_connection(
        self, modification: RemoveConnectionModification
    ) -> Optional[int]:
        if not hasattr(modification, "child_id"):
            return None
        for i, segment in enumerate(self.cable.segments):
            if segment.id == modification.child_id:
                if (
                    self.world.root is not None
                    and modification.parent_id == self.world.root.id
                ):
                    return None
                return i
        return None

    def start(self) -> None:
        """Start the background physics simulation thread."""
        if self._started:
            return
        self.multi_sim.start_simulation()
        self._started = True
        logger.info("Cable simulation started")

    def stop(self) -> None:
        """Stop the background physics simulation thread."""
        self._remove_position_override_hook()
        self._unregister_model_callback()
        if not self._started:
            return
        if self.multi_sim.simulator.state != SimulatorState.STOPPED:
            self.multi_sim.stop_simulation()
        self._started = False
        logger.info("Cable simulation stopped")

    def grasp(self, gripper_body_name: str, segment_index: int = 0) -> None:
        r"""
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
        strategy = self._effective_strategy
        if self.config.use_composite or strategy == CableSimulationStrategy.POSITION_OVERRIDE:
            self._grasp_via_position_override(gripper_body_name, segment_index)
        else:
            self._grasp_via_kinematic_attach(gripper_body_name, segment_index)

    def _grasp_via_kinematic_attach(
        self, gripper_body_name: str, segment_index: int
    ) -> None:
        r"""
        Attach the segment using the legacy kinematic reparenting,
        preserving the entire simulation state across recompilation.
        """
        import mujoco

        mj_data = self.multi_sim.simulator._mj_data
        saved_qpos = mj_data.qpos.copy()
        saved_qvel = mj_data.qvel.copy()

        segment_name = self._body_name_for_segment(segment_index)
        self.multi_sim.simulator.callbacks["attach"](
            body_1_name=segment_name,
            body_2_name=gripper_body_name,
        )

        mj_model = self.multi_sim.simulator._mj_model
        mj_data = self.multi_sim.simulator._mj_data

        if len(saved_qpos) == len(mj_data.qpos) and len(saved_qvel) == len(mj_data.qvel):
            mj_data.qpos[:] = saved_qpos
            mj_data.qvel[:] = saved_qvel

        logger.info(
            "Cable segment %d attached to %s", segment_index, gripper_body_name
        )

    def _grasp_via_position_override(
        self, gripper_body_name: str, segment_index: int
    ) -> None:
        r"""
        Store the grasp relationship and install a post-step qpos
        correction so the segment follows the gripper without changing
        the kinematic tree.
        """
        import mujoco

        mj_model = self.multi_sim.simulator._mj_model
        mj_data = self.multi_sim.simulator._mj_data

        gripper_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, gripper_body_name
        )
        if gripper_id == -1:
            raise ValueError(f"Gripper body '{gripper_body_name}' not found")

        segment_name = self._body_name_for_segment(segment_index)
        segment_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, segment_name
        )
        if segment_id == -1:
            raise ValueError(f"Segment body '{segment_name}' not found")

        gripper_joint_id = mj_model.body_jntadr[gripper_id]
        gripper_qpos_adr = mj_model.jnt_qposadr[gripper_joint_id]
        gripper_xpos = mj_data.qpos[gripper_qpos_adr : gripper_qpos_adr + 3].copy()
        gripper_xquat = mj_data.qpos[gripper_qpos_adr + 3 : gripper_qpos_adr + 7].copy()

        segment_joint_id = mj_model.body_jntadr[segment_id]
        segment_qpos_adr = mj_model.jnt_qposadr[segment_joint_id]
        segment_xpos = mj_data.qpos[segment_qpos_adr : segment_qpos_adr + 3].copy()
        segment_xquat = mj_data.qpos[segment_qpos_adr + 3 : segment_qpos_adr + 7].copy()

        gripper_neg_quat = numpy.zeros(4)
        mujoco.mju_negQuat(gripper_neg_quat, gripper_xquat)
        relative_position = segment_xpos - gripper_xpos
        mujoco.mju_rotVecQuat(relative_position, relative_position, gripper_neg_quat)
        relative_quaternion = numpy.zeros(4)
        mujoco.mju_mulQuat(relative_quaternion, gripper_neg_quat, segment_xquat)

        self._grasped_segments[segment_index] = (
            gripper_body_name,
            relative_position,
            relative_quaternion,
        )

        self._install_position_override_hook()
        logger.info(
            "Cable segment %d position-override grasped to %s",
            segment_index,
            gripper_body_name,
        )

    def _set_segment_qpos(
        self,
        segment_index: int,
        gripper_xpos: numpy.ndarray,
        gripper_xquat: numpy.ndarray,
        rel_pos: numpy.ndarray,
        rel_quat: numpy.ndarray,
    ) -> None:
        import mujoco

        mj_model = self.multi_sim.simulator._mj_model
        mj_data = self.multi_sim.simulator._mj_data

        segment_world_pos = numpy.zeros(3)
        mujoco.mju_rotVecQuat(segment_world_pos, rel_pos, gripper_xquat)
        segment_world_pos += gripper_xpos

        segment_world_quat = numpy.zeros(4)
        mujoco.mju_mulQuat(segment_world_quat, gripper_xquat, rel_quat)

        segment_name = self._body_name_for_segment(segment_index)
        segment_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, segment_name
        )
        if segment_id < 0:
            logger.warning("Segment body '%s' not in model", segment_name)
            return

        joint_id = mj_model.body_jntadr[segment_id]
        if joint_id < 0:
            logger.warning("Segment body '%s' has no joint", segment_name)
            return
        if mj_model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            logger.warning("Segment '%s' joint is not free", segment_name)
            return

        qpos_adr = mj_model.jnt_qposadr[joint_id]
        mj_data.qpos[qpos_adr : qpos_adr + 3] = segment_world_pos
        mj_data.qpos[qpos_adr + 3 : qpos_adr + 7] = segment_world_quat

    def release(self, segment_index: int = 0) -> None:
        r"""
        Detach a cable segment body from its parent in the running
        simulation, restoring gravity and constraint behaviour.

        :param segment_index: Index of the cable segment to release.
        """
        if not self._started:
            raise RuntimeError("Simulation is not running")
        if segment_index < 0 or segment_index >= len(self.cable.segments):
            raise ValueError(
                f"segment_index {segment_index} out of range "
                f" [0, {len(self.cable.segments)})"
            )
        strategy = self._effective_strategy
        if self.config.use_composite or strategy == CableSimulationStrategy.POSITION_OVERRIDE:
            self._release_via_position_override(segment_index)
        else:
            self._release_via_kinematic_detach(segment_index)

    def _release_via_kinematic_detach(self, segment_index: int) -> None:
        r"""
        Detach the segment using legacy kinematic reparenting,
        preserving the entire simulation state.
        """
        import mujoco

        mj_data = self.multi_sim.simulator._mj_data
        saved_qpos = mj_data.qpos.copy()
        saved_qvel = mj_data.qvel.copy()

        segment_name = self._body_name_for_segment(segment_index)
        self.multi_sim.simulator.callbacks["detach"](
            body_name=segment_name,
            add_freejoint=True,
        )

        mj_data = self.multi_sim.simulator._mj_data

        if len(saved_qpos) == len(mj_data.qpos) and len(saved_qvel) == len(mj_data.qvel):
            mj_data.qpos[:] = saved_qpos
            mj_data.qvel[:] = saved_qvel

        logger.info("Cable segment %d released", segment_index)

    def _release_via_position_override(self, segment_index: int) -> None:
        self._grasped_segments.pop(segment_index, None)
        if not self._grasped_segments:
            self._remove_position_override_hook()
        logger.info("Cable segment %d position-override released", segment_index)

    def _install_position_override_hook(self) -> None:
        if self._position_override_installed:
            return
        simulator = self.multi_sim.simulator
        self._original_step_callback = simulator.step_callback
        original_step = simulator.step_callback

        def step_with_override():
            self._apply_position_overrides()
            original_step()
            self._apply_position_overrides()

        simulator.step_callback = step_with_override
        self._position_override_installed = True

    def _remove_position_override_hook(self) -> None:
        if not self._position_override_installed:
            return
        simulator = self.multi_sim.simulator
        if self._original_step_callback is not None:
            simulator.step_callback = self._original_step_callback
            self._original_step_callback = None
        self._position_override_installed = False

    def _apply_position_overrides(self) -> None:
        r"""
        For every grasped segment, correct its free-joint qpos to keep
        it at the desired offset from the gripper.  Called before and
        after every simulation step while a segment is grasped.
        """
        if not self._grasped_segments:
            return
        import mujoco

        mj_model = self.multi_sim.simulator._mj_model
        mj_data = self.multi_sim.simulator._mj_data

        for segment_index, (
            gripper_name,
            rel_pos,
            rel_quat,
        ) in self._grasped_segments.items():
            gripper_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_BODY, gripper_name
            )
            if gripper_id == -1:
                continue

            gripper_joint_id = mj_model.body_jntadr[gripper_id]
            if gripper_joint_id < 0:
                continue

            gripper_qpos_adr = mj_model.jnt_qposadr[gripper_joint_id]
            gripper_xpos = mj_data.qpos[gripper_qpos_adr : gripper_qpos_adr + 3].copy()
            gripper_xquat = mj_data.qpos[gripper_qpos_adr + 3 : gripper_qpos_adr + 7].copy()

            self._set_segment_qpos(
                segment_index, gripper_xpos, gripper_xquat, rel_pos, rel_quat
            )

    def _body_name_for_segment(self, segment_index: int) -> str:
        if self.config.use_composite:
            return self._composite_body_names.get(segment_index, f"cable_segment_{segment_index}")
        return f"cable_segment_{segment_index}"

    def get_segment_positions(self) -> Dict[str, numpy.ndarray]:
        r"""
        Return the current world-frame position of every cable segment
        from the running simulation.  Reads qpos directly to avoid the
        one-step lag in cartesian positions (xpos).

        :return: Mapping ``{segment_name: numpy.ndarray([x, y, z]), ...}``
            where keys are the world-model body names (e.g.
            ``"cable_segment_0"``).
        """
        if not self._started:
            raise RuntimeError("Simulation is not running")
        import mujoco

        mj_model = self.multi_sim.simulator._mj_model
        mj_data = self.multi_sim.simulator._mj_data
        result: Dict[str, numpy.ndarray] = {}

        for i in range(len(self.cable.segments)):
            world_name = self.cable.segments[i].name.name
            mujoco_name = self._body_name_for_segment(i)
            body_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_BODY, mujoco_name
            )
            if body_id < 0:
                continue
            joint_id = mj_model.body_jntadr[body_id]
            if joint_id < 0:
                continue
            qpos_adr = mj_model.jnt_qposadr[joint_id]
            result[world_name] = mj_data.qpos[qpos_adr : qpos_adr + 3].copy()

        return result
