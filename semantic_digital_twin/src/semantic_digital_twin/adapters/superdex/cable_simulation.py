"""
Cables simulated as Cosserat rods by SuperDex, next to the rigid bodies of a world.

The rigid bodies move in another simulation, or kinematically; SuperDex sees each
collision shape of the bodies a cable may touch as a static collider it moves along with
its body, and the cable pushes nothing back.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from itertools import pairwise

import numpy
import superdex.physics as superdex_physics
from krrood.ormatic.utils import classproperty
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp
from typing_extensions import Any, Self

from semantic_digital_twin.adapters.superdex.exceptions import (
    BodyWithoutCollisionShapesError,
    CableNodeOutOfRangeError,
    CableSimulationDivergedError,
)
from semantic_digital_twin.semantic_annotations.deformable_cable import (
    DeformableCable,
    Polyline,
)
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.contact import (
    ContactFriction,
    ContactParameters,
)
from semantic_digital_twin.world_description.geometry import Box, Shape
from semantic_digital_twin.world_description.world_entity import Body

# %% contact


@dataclass
class ContactSettings:
    """
    How the cables' contacts with the colliders resolve.
    """

    penalty: float = 8.0e7
    """
    Stiffness of the penalty that pushes a cable out of a collider, in newtons per
    metre.
    """

    smoothing_distance: float = 5.0e-4
    """
    Half the distance, in metres, over which the penalty ramps up from zero; the smaller
    it is, the closer a cable rests on a surface.
    """

    skin_sides: int = 8
    """
    How many sides the tube has that a cable's contacts are sampled on.
    """


# %% moving bodies


@dataclass
class BodyPoses(ABC):
    """
    Where the bodies of a world are, as the cables see them.
    """

    @abstractmethod
    def world_T_body(self, body: Body) -> NDArray[numpy.float64]:
        """
        :param body: A body of the world.
        :return: Its pose now, a homogeneous transformation from the world root.
        """


@dataclass
class KinematicBodyPoses(BodyPoses):
    """
    The poses the state of a world puts its bodies in.
    """

    world: World
    """
    The world.
    """

    def world_T_body(self, body: Body) -> NDArray[numpy.float64]:
        return self.world.compute_forward_kinematics_np(self.world.root, body)


@dataclass
class BodyMotion:
    """
    How a body moved during one step of the simulation.
    """

    start: NDArray[numpy.float64]
    """
    The pose of the body, a homogeneous transformation from the world root, at the start
    of the step.
    """

    end: NDArray[numpy.float64]
    """
    Its pose at the end of the step.
    """

    def pose_at(self, fraction: float) -> NDArray[numpy.float64]:
        """
        :param fraction: How far through the step, from zero to one.
        :return: The pose of the body at that moment, moving at a constant linear and
            angular velocity through the step.
        """
        rotations = Rotation.from_matrix(
            numpy.stack([self.start[:3, :3], self.end[:3, :3]])
        )
        pose = numpy.eye(4)
        pose[:3, :3] = Slerp([0.0, 1.0], rotations)(fraction).as_matrix()
        pose[:3, 3] = (1.0 - fraction) * self.start[:3, 3] + fraction * self.end[:3, 3]
        return pose


# %% cables


@dataclass
class SimulatedCable:
    """
    A cable of the world and the SuperDex rod simulating it.
    """

    cable: DeformableCable
    """
    The cable of the world, whose nodes this rod updates.
    """

    actor: Any
    """
    The SuperDex rod actor.
    """

    reference_nodes: NDArray[numpy.float64]
    """
    The nodes the rod was built with, of shape ``(number of nodes, 3)``; the rod states
    its nodes as displacements from these.
    """

    degrees_of_freedom_per_node: int
    """
    How many degrees of freedom the rod has per node: the three translations and the
    twist of the segment after it.
    """

    @classmethod
    def build(
        cls, scene: Any, cable: DeformableCable, contact_settings: ContactSettings
    ) -> Self:
        """
        Build the rod of a cable in a scene, straight when unloaded and laid along the
        cable's nodes.

        :param scene: The SuperDex scene to build the rod in.
        :param cable: The cable to simulate.
        :param contact_settings: How its contacts resolve.
        :return: The simulated cable.
        """
        start_nodes = cable.centreline.corners
        straight_nodes = cls._straightened(start_nodes)
        shape = superdex_physics.create_model_shape(
            superdex_physics.experimental.generate_tubular_rod_model_data(
                nodes=straight_nodes.astype(numpy.float32),
                element_frame_axes=cls._material_frame_axes(straight_nodes).astype(
                    numpy.float32
                ),
                radius=cable.radius,
                num_cross_section_segments=contact_settings.skin_sides,
                is_closed_loop=False,
            )
        )
        material = superdex_physics.experimental.RodMaterialParams()
        material.axial_stiffness = cable.axial_stiffness
        material.flexural_stiffness = [cable.bending_stiffness] * 2
        material.torsional_stiffness = cable.torsional_stiffness
        material.linear_density = cable.linear_density
        material.linear_rotational_inertia = cable.rotational_inertia_per_length
        material.stiffness_damping_coefficient = cable.material.stiffness_damping

        parameters = superdex_physics.experimental.RodActorParams()
        parameters.name = cable.name.name
        parameters.shape = shape
        parameters.material = material
        parameters.world_from_local = superdex_physics.TransformRT()
        parameters.has_gravity = True
        parameters.collider_type = superdex_physics.ColliderType.POINT_CLOUD
        parameters.point_cloud_collider.radius = cable.radius
        parameters.point_cloud_collider.self_contact = False
        # Contact is sampled on the tubular skin, so the cable rests on its surface.
        parameters.use_visual_mesh_contact = True
        parameters.contact = superdex_physics.ContactParams(
            penalty_coefficient=contact_settings.penalty,
            coulomb_friction_coefficient=cable.material.friction_coefficient,
            friction_falloff_vel=cable.material.friction_falloff_velocity,
        )
        actor = superdex_physics.experimental.create_rod_actor(scene, parameters)
        reference_nodes = numpy.asarray(
            actor.get_mesh().coordinates, dtype=numpy.float64
        ).reshape(-1, 3)
        return cls(
            cable=cable,
            actor=actor,
            reference_nodes=reference_nodes,
            degrees_of_freedom_per_node=len(actor.get_displacements())
            // len(reference_nodes),
        )

    @staticmethod
    def _straightened(nodes: NDArray[numpy.float64]) -> NDArray[numpy.float64]:
        """
        :param nodes: The nodes of a centreline.
        :return: The nodes of a straight centreline with the same segment lengths, from
            the same first node along the first segment.
        """
        direction = nodes[1] - nodes[0]
        direction /= numpy.linalg.norm(direction)
        arc_lengths = Polyline(nodes).arc_lengths
        return nodes[0] + arc_lengths[:, None] * direction[None, :]

    @staticmethod
    def _material_frame_axes(nodes: NDArray[numpy.float64]) -> NDArray[numpy.float64]:
        """
        One axis perpendicular to each segment, carried from segment to segment by the
        rotation between their directions, so the rod starts without twist.

        :param nodes: The nodes of a centreline.
        :return: The axes, of shape ``(number of segments, 3)``.
        """
        tangents = numpy.diff(nodes, axis=0)
        tangents /= numpy.linalg.norm(tangents, axis=1, keepdims=True)
        helper = (
            numpy.array([0.0, 0.0, 1.0])
            if abs(tangents[0][2]) < 0.9
            else numpy.array([1.0, 0.0, 0.0])
        )
        axis = numpy.cross(tangents[0], helper)
        axis /= numpy.linalg.norm(axis)
        axes = [axis]
        for previous, current in pairwise(tangents):
            rotation_axis = numpy.cross(previous, current)
            sine = numpy.linalg.norm(rotation_axis)
            if sine > 1.0e-12:
                angle = numpy.arctan2(sine, numpy.dot(previous, current))
                axis = Rotation.from_rotvec(rotation_axis / sine * angle).apply(axis)
            axis -= numpy.dot(axis, current) * current
            axis /= numpy.linalg.norm(axis)
            axes.append(axis)
        return numpy.array(axes)

    @property
    def node_positions(self) -> NDArray[numpy.float64]:
        """
        Where the rod's nodes are now, of shape ``(number of nodes, 3)``.
        """
        displacements = numpy.asarray(self.actor.get_displacements()).reshape(
            -1, self.degrees_of_freedom_per_node
        )
        return self.reference_nodes + displacements[:, :3]

    def hold_at(self, positions: NDArray[numpy.float64]) -> None:
        """
        Hold every node of the rod at a position until :meth:`let_go`.

        :param positions: Where to hold the nodes, of shape ``(number of nodes, 3)``.
        """
        node_indices = numpy.arange(len(positions))
        degrees_of_freedom = (
            node_indices[:, None] * self.degrees_of_freedom_per_node
            + numpy.arange(3)[None, :]
        ).reshape(-1)
        self.actor.clear_boundary_conditions()
        self.actor.add_boundary_condition_dofs_world(
            dof_indices=degrees_of_freedom.astype(numpy.int32),
            dof_values_world=numpy.asarray(positions, dtype=numpy.float64).reshape(-1),
        )

    def let_go(self) -> None:
        """
        Release the nodes :meth:`hold_at` held and let them rest.
        """
        self.actor.clear_boundary_conditions()
        self.actor.set_node_velocities_local(
            numpy.zeros(len(self.actor.get_displacements()))
        )

    def publish(self) -> None:
        """
        Write where the rod's nodes are now into the cable of the world.
        """
        reference_frame = self.cable.nodes[0].reference_frame
        self.cable.nodes = [
            Point3.from_iterable(position, reference_frame=reference_frame)
            for position in self.node_positions
        ]


# %% colliders


@dataclass
class SimulatedCollider:
    """
    A collision shape of a body, as a static SuperDex collider that moves with the
    body.
    """

    body: Body
    """
    The body the shape belongs to.
    """

    body_T_shape: NDArray[numpy.float64]
    """
    The pose of the shape relative to its body.
    """

    actor: Any
    """
    The SuperDex rigid actor.
    """

    @classmethod
    def build(
        cls,
        scene: Any,
        body: Body,
        shape: Shape,
        world_T_body: NDArray[numpy.float64],
        contact_settings: ContactSettings,
    ) -> Self:
        """
        Build the collider of a collision shape in a scene.

        A box becomes an exact box collider, every other shape a signed distance field
        of its mesh.

        :param scene: The SuperDex scene to build the collider in.
        :param body: The body the shape belongs to.
        :param shape: The collision shape.
        :param world_T_body: Where the body is.
        :param contact_settings: How contacts with the collider resolve.
        :return: The collider.
        """
        world = body._world
        body_T_shape = world.transform(shape.origin, body).to_np()
        contact_parameters = shape.get_simulator_property_of_type(ContactParameters)
        friction = (
            contact_parameters.friction
            if contact_parameters is not None
            else ContactFriction()
        )
        mesh = shape.mesh
        collider = cls(
            body=body,
            body_T_shape=body_T_shape,
            actor=scene.create_rigid_actor(
                name=str(body.name),
                layer="collider",
                shape=superdex_physics.create_tri_mesh_shape(
                    coordinates=numpy.asarray(
                        mesh.vertices, dtype=numpy.float64
                    ).ravel(),
                    connectivity=numpy.asarray(mesh.faces, dtype=numpy.int32).ravel(),
                ),
                is_static=True,
                collider_type=(
                    superdex_physics.ColliderType.BOX
                    if isinstance(shape, Box)
                    else superdex_physics.ColliderType.SDF
                ),
                world_from_local=superdex_physics.TransformRT(),
                contact=superdex_physics.ContactParams(
                    coulomb_friction_coefficient=friction.sliding,
                    penalty_smoothing_half_distance=contact_settings.smoothing_distance,
                ),
            ),
        )
        collider.move_to(world_T_body)
        return collider

    def move_to(self, world_T_body: NDArray[numpy.float64]) -> None:
        """
        Move the collider to where its shape is when its body is at a pose.

        :param world_T_body: The pose of the body.
        """
        world_T_shape = world_T_body @ self.body_T_shape
        x, y, z, w = Rotation.from_matrix(world_T_shape[:3, :3]).as_quat()
        self.actor.set_root_transform(
            superdex_physics.TransformRT(
                rotation=superdex_physics.Quaternion(x, y, z, w),
                translation=world_T_shape[:3, 3].tolist(),
            )
        )


# %% pins


@dataclass
class CablePin:
    """
    A node of a cable held at a point fixed to a body, by a stiff spring.
    """

    cable: SimulatedCable
    """
    The cable whose node is held.
    """

    node_index: int
    """
    The index of the held node.
    """

    body: Body
    """
    The body the node is held to.
    """

    body_P_node: NDArray[numpy.float64]
    """
    Where on the body the node is held, in the body's frame.
    """

    constraint: Any
    """
    The SuperDex constraint holding the node.
    """

    def aim(self, world_T_body: NDArray[numpy.float64]) -> None:
        """
        Hold the node where its point is when the body is at a pose.

        :param world_T_body: The pose of the body.
        """
        target = world_T_body[:3, :3] @ self.body_P_node + world_T_body[:3, 3]
        self.constraint.set_target_position(target.tolist())


# %% simulation


@dataclass
class SuperdexCableSimulation:
    """
    Simulates every :class:`~semantic_digital_twin.semantic_annotations.deformable_cable.DeformableCable`
    of a world, touching the collision shapes of a chosen set of bodies.

    Each :meth:`step` sweeps the bodies from where they were at the previous step to
    where they are now.
    """

    world: World
    """
    The world whose cables are simulated and whose bodies they touch.
    """

    colliding_bodies: list[Body]
    """
    The bodies whose collision shapes the cables touch.
    """

    body_poses: BodyPoses
    """
    Where the bodies are.
    """

    maximum_substep_duration: timedelta = timedelta(milliseconds=10)
    """
    The longest step the cable solver takes; a :meth:`step` is split into as many
    equal steps as this needs.
    """

    contact_settings: ContactSettings = field(default_factory=ContactSettings)
    """
    How contacts with the colliding bodies resolve.
    """

    pin_stiffness: float = 2000.0
    """
    Stiffness of the spring a :class:`CablePin` holds its node with, in newtons per
    metre.
    """

    pin_damping: float = 10.0
    """
    Damping of that spring, in newton seconds per metre.
    """

    gravitational_acceleration: float = 9.81
    """
    How fast the cables fall along the world's negative z axis, in metres per second
    squared.
    """

    worker_threads: int = -1
    """
    How many threads SuperDex solves with; ``-1`` lets it decide. Only the first
    simulation of a process decides, because the engine starts once per process.
    """

    scene: Any = field(init=False)
    """
    The SuperDex scene.
    """

    cables: list[SimulatedCable] = field(init=False, default_factory=list)
    """
    The simulated cables.
    """

    colliders: list[SimulatedCollider] = field(init=False, default_factory=list)
    """
    The colliders of the colliding bodies' collision shapes.
    """

    pins: list[CablePin] = field(init=False, default_factory=list)
    """
    The nodes held to bodies.
    """

    _previous_world_T_bodies: dict[Body, NDArray[numpy.float64]] = field(
        init=False, default_factory=dict
    )
    """
    The poses of the colliding and pinned bodies at the end of the previous step.
    """

    @classproperty
    def placement_step_duration(cls) -> timedelta:
        """
        The step that carries a newly built rod onto its starting nodes; short enough
        that nothing else moves in it.
        """
        return timedelta(microseconds=1)

    def __post_init__(self):
        if not superdex_physics.is_initialized():
            superdex_physics.initialize(num_worker_threads=self.worker_threads)
        self.scene = superdex_physics.create_scene(str(self.world.name))
        self.scene.set_gravity([0.0, 0.0, -self.gravitational_acceleration])
        for cable in self.world.get_semantic_annotations_by_type(DeformableCable):
            self._add_cable(cable)
        for body in self.colliding_bodies:
            self._add_colliders_of(body)

    def _add_cable(self, cable: DeformableCable) -> None:
        """
        Simulate a cable, carrying its rod from its straight rest shape onto the cable's
        nodes and holding every cable built before it in place meanwhile.

        :param cable: The cable to simulate.
        """
        simulated_cable = SimulatedCable.build(self.scene, cable, self.contact_settings)
        held = [(other, other.node_positions) for other in self.cables] + [
            (simulated_cable, cable.centreline.corners)
        ]
        gravity = list(self.scene.get_gravity())
        self.scene.set_gravity([0.0, 0.0, 0.0])
        # The first tiny step moves the nodes onto their targets, the second takes
        # the speed the first gave them away again.
        for _ in range(2):
            for held_cable, positions in held:
                held_cable.hold_at(positions)
            self.scene.step(self.placement_step_duration.total_seconds())
        self.scene.set_gravity(gravity)
        for held_cable, _ in held:
            held_cable.let_go()
        self.cables.append(simulated_cable)

    def _add_colliders_of(self, body: Body) -> None:
        """
        Build a collider for every collision shape of a body.

        :param body: The body.
        :raises BodyWithoutCollisionShapesError: If the body has no collision shapes.
        """
        if len(body.collision) == 0:
            raise BodyWithoutCollisionShapesError(body=body)
        world_T_body = self.body_poses.world_T_body(body)
        self._previous_world_T_bodies[body] = world_T_body
        for shape in body.collision:
            self.colliders.append(
                SimulatedCollider.build(
                    self.scene, body, shape, world_T_body, self.contact_settings
                )
            )

    def simulated_cable_of(self, cable: DeformableCable) -> SimulatedCable:
        """
        :param cable: A cable of the world.
        :return: Its simulation.
        """
        return next(simulated for simulated in self.cables if simulated.cable is cable)

    def step(self, duration: timedelta) -> None:
        """
        Advance the cables by a period, sweeping the colliders and pins from where
        their bodies were at the previous step to where they are now, and write the
        cables' new nodes into the world.

        :param duration: How much simulated time to advance.
        :raises CableSimulationDivergedError: If the solver diverged.
        """
        motions = {
            body: BodyMotion(start=start, end=self.body_poses.world_T_body(body))
            for body, start in self._previous_world_T_bodies.items()
        }
        substeps = math.ceil(duration / self.maximum_substep_duration)
        substep_seconds = duration.total_seconds() / substeps
        for substep in range(substeps):
            fraction = (substep + 1) / substeps
            poses = {body: motion.pose_at(fraction) for body, motion in motions.items()}
            for collider in self.colliders:
                collider.move_to(poses[collider.body])
            for pin in self.pins:
                pin.aim(poses[pin.body])
            self.scene.step(substep_seconds)
        statistics = self.scene.get_solver_stats()
        if "DIVERGED" in str(statistics.convergence_status):
            raise CableSimulationDivergedError(residual=float(statistics.residual_norm))
        self._previous_world_T_bodies = {
            body: motion.end for body, motion in motions.items()
        }
        for cable in self.cables:
            cable.publish()

    def pin(self, cable: DeformableCable, node_index: int, body: Body) -> CablePin:
        """
        Hold a node of a cable where it is now, relative to a body, until
        :meth:`release`.

        :param cable: The cable.
        :param node_index: The index of the node to hold.
        :param body: The body to hold it to.
        :return: The pin holding the node.
        :raises CableNodeOutOfRangeError: If the cable has no such node.
        """
        simulated_cable = self.simulated_cable_of(cable)
        positions = simulated_cable.node_positions
        if not 0 <= node_index < len(positions):
            raise CableNodeOutOfRangeError(
                node_index=node_index, number_of_nodes=len(positions)
            )
        world_T_body = self.body_poses.world_T_body(body)
        self._previous_world_T_bodies.setdefault(body, world_T_body)
        node_position = positions[node_index]
        pin = CablePin(
            cable=simulated_cable,
            node_index=node_index,
            body=body,
            body_P_node=world_T_body[:3, :3].T @ (node_position - world_T_body[:3, 3]),
            constraint=self.scene.create_deformable_node_position_constraint(
                stiffness=self.pin_stiffness,
                damping=self.pin_damping,
                saturation=-1.0,
                node_index=node_index,
                position=node_position.tolist(),
                actor=simulated_cable.actor.get_handle(),
            ),
        )
        self.pins.append(pin)
        return pin

    def release(self, pin: CablePin) -> None:
        """
        Let go of a node :meth:`pin` held.

        :param pin: The pin to release.
        """
        self.scene.destroy_constraint(pin.constraint)
        self.pins.remove(pin)
