import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, List

import coacd
import numpy as np
import trimesh
from pygments.styles import vs

from semantic_digital_twin.pipeline.pipeline import Step
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Shape, Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


class ApproximationMode(StrEnum):
    """
    Approximation shape type
    """

    BOX = "box"
    CONVEX_HULL = "ch"


class PreprocessingMode(StrEnum):
    """
    Manifold preprocessing mode
    """

    AUTO = "auto"
    """
    Automatically chose based on the geometry.
    """

    ON = "on"
    """
    Force turn on the pre-processing
    """

    OFF = "off"
    """
    Force turn off the pre-processing
    """


class FillMode(StrEnum):
    """
    The Fill Mode determines how V-HACD "solidifies" your mesh before it starts breaking it into convex pieces.
    """

    FLOOD = "flood"
    """
    Fills the volume from the outside in.
    Best for Solid, closed (watertight) objects.
    """

    SURFACE = "surface"
    """
    Only processes the skin of the mesh.
    Best for hollow objects or "open" meshes (like a plane).
    """

    RAYCAST = "raycast"
    """
    Checks interior by shooting rays.
    Best for messy geometry with holes or overlapping parts.
    """


@dataclass
class MeshDecomposer(Step, ABC):
    """
    MeshDecomposer is an abstract base class for decomposing complex 3D meshes into simpler convex components.
    It provides methods to apply the decomposition to meshes, shapes, bodies, and entire worlds.
    Subclasses should implement the `apply_to_mesh` method to define the specific decomposition algorithm.
    3D meshes are represented using the `trimesh` library, and the decomposed parts are returned as a list of `TriangleMesh` objects.
    """

    @abstractmethod
    def apply_to_mesh(self, mesh: Mesh) -> List[Mesh]:
        """
        Apply the mesh decomposition to a given mesh.
        Returns a list of TriangleMesh objects representing the decomposed convex parts.
        """
        ...

    def apply_to_shape(self, shape: Shape) -> List[Mesh]:
        """
        Apply the mesh decomposition to a given shape.
        If the shape is a Mesh, it will be decomposed into multiple TriangleMesh objects.
        Otherwise, the shape will be returned as is in a list.
        """
        new_geometry = []
        if isinstance(shape, Mesh):
            new_geometry = self.apply_to_mesh(shape)
        else:
            new_geometry.append(shape)

        return new_geometry

    def apply_to_body(self, body: Body) -> Body:
        """
        Apply the mesh decomposition to all shapes in a given body.
        The body's collision shapes will be replaced with the decomposed shapes.
        Returns the modified body.
        """
        new_geometry = []
        for shape in body.visual:
            decomposed_shapes = self.apply_to_shape(shape)
            new_geometry.extend(decomposed_shapes)

        body.collision = ShapeCollection(new_geometry)
        return body

    def _apply(self, world: World) -> World:
        """
        Apply the mesh decomposition to all bodies in a given world.
        Each body's collision shapes will be replaced with the decomposed shapes.
        Returns the modified world.
        """
        for body in world.bodies:
            self.apply_to_body(body)

        return world


@dataclass
class COACDMeshDecomposer(MeshDecomposer):
    """
    COACDMeshDecomposer is a class for decomposing complex 3D meshes into simpler convex components
    using the COACD (Convex Optimization for Approximate Convex Decomposition) algorithm. It is
    designed to preprocess, analyze, and process 3D meshes with a focus on efficiency and scalability
    in fields such as robotics, gaming, and simulation.

    Check https://github.com/SarahWeiii/CoACD for further details.
    """

    threshold: float = 0.05
    """
    Determines how much the decomposed parts can deviate from the original shape. 
    A lower value leads to more pieces and higher precision, while a higher value allows for a 
    "looser," simpler approximation.
    """

    max_convex_hull: Optional[int] = None
    """
    Limits the total number of convex pieces generated. 
    If set to None, the algorithm will keep creating pieces until the threshold is met.
    Works only when merge is enabled (may introduce convex hull with a concavity larger than the threshold)
    """

    preprocess_mode: PreprocessingMode = PreprocessingMode.AUTO
    """
    Determines if the mesh is cleaned before processing.
    """

    preprocess_resolution: int = 50
    """
    The grid size used during the initial cleanup phase. 
    Low values are fast but might "melt" away small features of 
    your model before the main decomposition even starts.
    Range: 20 - 100
    """

    resolution: int = 2000
    """
    Defines the sampling density of the input mesh. 
    Think of it like the "DPI" of a printer; a higher resolution captures 
    finer details of the surface but increases the time it takes to calculate the decomposition.
    Range: 1 000 - 10 000
    """

    search_nodes: int = 20
    """
    Max number of child nodes in the monte carlo tree search (10 - 40).
    """

    search_iterations: int = 150
    """
    Number of search iterations in the monte carlo tree search (60 - 2000).
    """

    search_depth: int = 3
    """
    Maximum search depth in the monte carlo tree search (2 - 7).
    """

    pca: bool = False
    """
    Enable PCA pre-processing.
    Stands for Principal Component Analysis. 
    If enabled, it uses the orientation of the object’s volume to align the decomposition cuts. 
    It’s great for long, thin objects but can sometimes ignore local symmetry.
    """

    merge: bool = True
    """
    Enable merge postprocessing.
    After splitting the mesh, 
    the algorithm checks if any adjacent pieces are "convex enough" to be joined back together. 
    This helps keep the final count of pieces low without sacrificing much accuracy.
    """

    max_convex_hull_vertices: Optional[int] = None
    """
    Maximum vertex value for each convex hull, only when decimate is enabled.
    Limits how many "corners" (vertices) each individual convex piece can have. 
    This is vital for physics engines (like Bullet or PhysX) which often have a hard limit (e.g., 32 or 64 vertices)
    for real-time performance.
    """

    extrude_margin: Optional[float] = None
    """
    Extrude margin, only when extrude is enabled.
    Adds a small "padding" or thickness to the generated pieces. 
    Useful if you're experiencing "tunneling" 
    in physics simulations where objects pass through each other.
    """

    approximation_mode: ApproximationMode = ApproximationMode.BOX
    """
    Defines the shape of the primitives used during the search phase.
    """

    seed: int = field(default_factory=lambda: np.random.randint(2**32))
    """
    Random seed used for sampling.
    """

    def apply_to_mesh(self, mesh: Mesh) -> List[Mesh]:
        """
        Apply the COACD mesh decomposition to a given mesh.
        Returns a list of TriangleMesh objects representing the decomposed convex parts.
        """
        new_geometry = []

        trimesh_mesh = mesh.mesh
        if mesh.scale.x == mesh.scale.y == mesh.scale.z:
            trimesh_mesh.apply_scale(mesh.scale.x)
        else:
            logging.warning("Ambiguous scale for mesh, using uniform scale only.")

        coacd_mesh = coacd.Mesh(trimesh_mesh.vertices, trimesh_mesh.faces)
        if self.max_convex_hull is not None:
            max_convex_hull = self.max_convex_hull
        else:
            max_convex_hull = -1
        parts = coacd.run_coacd(
            mesh=coacd_mesh,
            apx_mode=str(self.approximation_mode),
            threshold=self.threshold,
            max_convex_hull=max_convex_hull,
            preprocess_mode=str(self.preprocess_mode),
            resolution=self.resolution,
            mcts_nodes=self.search_nodes,
            mcts_iterations=self.search_iterations,
            mcts_max_depth=self.search_depth,
            pca=self.pca,
            merge=self.merge,
            decimate=self.max_convex_hull_vertices is not None,
            max_ch_vertex=self.max_convex_hull_vertices or 256,
            extrude=self.extrude_margin is not None,
            extrude_margin=self.extrude_margin or 0.01,
            seed=self.seed,
        )

        for vs, fs in parts:
            new_geometry.append(
                Mesh.from_trimesh(mesh=trimesh.Trimesh(vs, fs), origin=mesh.origin)
            )

        return new_geometry


@dataclass
class VHACDMeshDecomposer(MeshDecomposer):
    """
    Decompose meshes using Voxelized Hierarchical Approximate Convex Decomposition.
    Read more about it here: https://pypi.org/project/vhacdx/
    """

    max_convex_hulls: int = 64
    """
    The maximum number of separate convex pieces the algorithm is allowed to generate.
    
    Originally referenced as maxConvexHulls.
    """

    resolution: int = 400000
    """
    The number of "voxels" (3D pixels) used to represent the shape during processing; 
    higher values capture finer details but take longer to compute.
    """

    minimum_volume_percent_error_allowed: float = 1.0
    """
    The threshold for how much the final decomposition can vary in volume from the original shape before the algorithm stops refining.    
    
    Originally referenced as minimumVolumePercentErrorAllowed.
    """

    max_recursion_depth: int = 10
    """
    How many times the algorithm can split a shape into smaller sub-parts to find a better fit.
    
    Originally referenced as maxRecursionDepth.
    """

    shrink_wrap: bool = True
    """
    When enabled, the algorithm "shrinks" the convex hulls to better wrap around the surface of the original mesh.
    
    Originally referenced as shrinkWrap.
    """

    fill_mode: FillMode = FillMode.FLOOD
    """
    Determines how the inside of the mesh is identified; "flood" fills the interior starting from the outside 
    (useful for solid objects).
    
    Originally referenced as fillMode.
    """

    max_vertices_per_convex_hull: int = 64
    """
    The maximum number of points allowed for each individual convex hull generated.
    
    Originally referenced as maxNumVerticesPerCH.
    """

    asynchronous: bool = True
    """
    If true, the decomposition runs in a separate thread, preventing the main program from freezing during calculation.
    
    Originally referenced as asyncACD.
    """

    min_edge_length: int = 2
    """
    The minimum size an edge must be to be considered during the simplification process.
    
    Originally referenced as minEdgeLength.
    """

    find_best_plane: bool = False
    """
    Tells the algorithm to search for the mathematically optimal plane when splitting a shape into two parts.
    
    Originally referenced as findBestPlane.
    """

    def apply_to_mesh(self, mesh: Mesh) -> List[Mesh]:
        decomposed = mesh.mesh.convex_decomposition(
            maxConvexHulls=self.max_convex_hulls,
            resolution=self.resolution,
            minimumVolumePercentErrorAllowed=self.minimum_volume_percent_error_allowed,
            maxRecursionDepth=self.max_recursion_depth,
            shrinkWrap=self.shrink_wrap,
            fillMode=self.fill_mode.value,
            maxNumVerticesPerCH=self.max_vertices_per_convex_hull,
            asyncACD=self.asynchronous,
            minEdgeLength=self.min_edge_length,
            findBestPlane=self.find_best_plane,
        )
        new_geometry = [
            Mesh.from_trimesh(mesh=decomposed_part, origin=mesh.origin)
            for decomposed_part in decomposed
        ]
        return new_geometry
