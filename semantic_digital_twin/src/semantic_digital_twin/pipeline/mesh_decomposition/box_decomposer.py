from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import trimesh
from scipy import ndimage
import time

from semantic_digital_twin.pipeline.mesh_decomposition.base import MeshDecomposer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Mesh, Box, Scale


@dataclass(frozen=True)
class GiangBox:
    position: list[float]
    size: list[float]


@dataclass(frozen=True)
class IndexBox:
    x0: int
    x1: int
    y0: int
    y1: int
    z0: int
    z1: int

    def dims(self) -> tuple[int, int, int]:
        return (self.x1 - self.x0, self.y1 - self.y0, self.z1 - self.z0)

    def volume_vox(self) -> int:
        dx, dy, dz = self.dims()
        return max(0, dx) * max(0, dy) * max(0, dz)

    def thin_axis(self) -> int:
        dx, dy, dz = self.dims()
        return int(np.argmin([dx, dy, dz]))

    def thickness_vox(self) -> int:
        dx, dy, dz = self.dims()
        return int(min(dx, dy, dz))

    def planar_area_vox(self) -> int:
        dx, dy, dz = sorted(self.dims())
        return int(dy * dz)


def index_box_to_box(idx: IndexBox, pitch: float, origin: np.ndarray) -> GiangBox:
    mins = origin + pitch * np.array([idx.x0, idx.y0, idx.z0], dtype=float)
    maxs = origin + pitch * np.array([idx.x1, idx.y1, idx.z1], dtype=float)
    return GiangBox(
        position=((mins + maxs) / 2.0).tolist(),
        size=((maxs - mins) / 2.0).tolist(),
    )


def greedy_merge_boxes(
    occupancy: np.ndarray, pitch: float, origin: np.ndarray
) -> list[GiangBox]:
    occ = occupancy.copy()
    nx, ny, nz = occ.shape
    boxes: list[GiangBox] = []

    for z in range(nz):
        for y in range(ny):
            x = 0
            while x < nx:
                if not occ[x, y, z]:
                    x += 1
                    continue

                x1 = x
                while x1 + 1 < nx and occ[x1 + 1, y, z]:
                    x1 += 1

                y1 = y
                while y1 + 1 < ny and occ[x : x1 + 1, y1 + 1, z].all():
                    y1 += 1

                z1 = z
                while z1 + 1 < nz and occ[x : x1 + 1, y : y1 + 1, z1 + 1].all():
                    z1 += 1

                occ[x : x1 + 1, y : y1 + 1, z : z1 + 1] = False

                mins = origin + pitch * np.array([x, y, z], dtype=float)
                maxs = origin + pitch * np.array([x1 + 1, y1 + 1, z1 + 1], dtype=float)
                boxes.append(
                    GiangBox(
                        position=((mins + maxs) / 2.0).tolist(),
                        size=((maxs - mins) / 2.0).tolist(),
                    )
                )

                x = x1 + 1

    return boxes


def clean_occupancy(occupancy: np.ndarray, fill_thin_holes: bool = True) -> np.ndarray:
    occ = occupancy.copy().astype(bool)
    occ = ndimage.binary_fill_holes(occ)

    if not fill_thin_holes:
        return occ.astype(bool)

    empty = ~occ
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, num = ndimage.label(empty, structure=structure)
    if num == 0:
        return occ.astype(bool)

    slices = ndimage.find_objects(labels)
    for lab, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        sx, sy, sz = slc
        dx = sx.stop - sx.start
        dy = sy.stop - sy.start
        dz = sz.stop - sz.start

        # Optional crack fill; disable with --no-thin-hole-fill if it fattens boards too much.
        if dx <= 1 or dy <= 1 or dz <= 1:
            occ[labels == lab] = True

    return occ.astype(bool)


def add_candidate(
    candidates: list[IndexBox],
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    z0: int,
    z1: int,
) -> None:
    if x1 > x0 and y1 > y0 and z1 > z0:
        candidates.append(IndexBox(x0, x1, y0, y1, z0, z1))


def fill_ratio_2d(mask2d: np.ndarray) -> float:
    if mask2d.size == 0:
        return 0.0
    return float(mask2d.mean())


def detect_planar_boards(
    occupancy: np.ndarray,
    max_thickness_vox: int = 2,
    min_span_vox: int = 3,
    min_fill_ratio: float = 0.75,
) -> list[IndexBox]:
    occ = occupancy
    nx, ny, nz = occ.shape
    candidates: list[IndexBox] = []

    # thin in X -> YZ boards
    for x0 in range(nx):
        for t in range(1, max_thickness_vox + 1):
            x1 = x0 + t
            if x1 > nx:
                continue
            slab = occ[x0:x1, :, :]
            if not slab.any():
                continue

            proj = slab.any(axis=0)  # (ny, nz)
            labels, num = ndimage.label(proj)
            for lab in range(1, num + 1):
                m = labels == lab
                ys, zs = np.where(m)
                if ys.size == 0:
                    continue
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                z0, z1 = int(zs.min()), int(zs.max()) + 1

                if (y1 - y0) < min_span_vox or (z1 - z0) < min_span_vox:
                    continue
                if fill_ratio_2d(m[y0:y1, z0:z1]) < min_fill_ratio:
                    continue

                add_candidate(candidates, x0, x1, y0, y1, z0, z1)

    # thin in Y -> XZ boards
    for y0 in range(ny):
        for t in range(1, max_thickness_vox + 1):
            y1 = y0 + t
            if y1 > ny:
                continue
            slab = occ[:, y0:y1, :]
            if not slab.any():
                continue

            proj = slab.any(axis=1)  # (nx, nz)
            labels, num = ndimage.label(proj)
            for lab in range(1, num + 1):
                m = labels == lab
                xs, zs = np.where(m)
                if xs.size == 0:
                    continue
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                z0, z1 = int(zs.min()), int(zs.max()) + 1

                if (x1 - x0) < min_span_vox or (z1 - z0) < min_span_vox:
                    continue
                if fill_ratio_2d(m[x0:x1, z0:z1]) < min_fill_ratio:
                    continue

                add_candidate(candidates, x0, x1, y0, y1, z0, z1)

    # thin in Z -> XY boards
    for z0 in range(nz):
        for t in range(1, max_thickness_vox + 1):
            z1 = z0 + t
            if z1 > nz:
                continue
            slab = occ[:, :, z0:z1]
            if not slab.any():
                continue

            proj = slab.any(axis=2)  # (nx, ny)
            labels, num = ndimage.label(proj)
            for lab in range(1, num + 1):
                m = labels == lab
                xs, ys = np.where(m)
                if xs.size == 0:
                    continue
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1

                if (x1 - x0) < min_span_vox or (y1 - y0) < min_span_vox:
                    continue
                if fill_ratio_2d(m[x0:x1, y0:y1]) < min_fill_ratio:
                    continue

                add_candidate(candidates, x0, x1, y0, y1, z0, z1)

    return candidates


def intersection_volume_vox(a: IndexBox, b: IndexBox) -> int:
    dx = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
    dy = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
    dz = max(0, min(a.z1, b.z1) - max(a.z0, b.z0))
    return dx * dy * dz


def deduplicate_index_boxes(
    candidates: list[IndexBox],
    overlap_ratio_threshold: float = 0.8,
) -> list[IndexBox]:
    """
    Prefer board candidates with:
    1) larger planar area
    2) thinner thickness
    3) then larger volume

    This avoids keeping a too-thick candidate when a thinner candidate covers
    the same board.
    """
    ordered = sorted(
        candidates,
        key=lambda b: (
            -b.planar_area_vox(),  # prefer large boards
            b.thickness_vox(),  # prefer thinner boards
            -b.volume_vox(),  # tie-breaker
            b.x0,
            b.y0,
            b.z0,
        ),
    )
    kept: list[IndexBox] = []

    for cand in ordered:
        cand_vol = cand.volume_vox()
        if cand_vol == 0:
            continue

        discard = False
        for prev in kept:
            inter = intersection_volume_vox(cand, prev)

            # If most of this candidate is already covered, drop it.
            if inter / cand_vol >= overlap_ratio_threshold:
                discard = True
                break

            # Extra rule: if same planar support but cand is thicker and overlaps strongly,
            # drop the thicker one.
            same_thin_axis = cand.thin_axis() == prev.thin_axis()
            similar_planar_area = (
                min(cand.planar_area_vox(), prev.planar_area_vox())
                / max(cand.planar_area_vox(), prev.planar_area_vox())
                >= 0.9
            )
            if (
                same_thin_axis
                and similar_planar_area
                and cand.thickness_vox() >= prev.thickness_vox()
            ):
                if inter / min(cand_vol, prev.volume_vox()) >= 0.6:
                    discard = True
                    break

        if not discard:
            kept.append(cand)

    return kept


def subtract_index_boxes_from_occupancy(
    occupancy: np.ndarray, boxes: list[IndexBox]
) -> np.ndarray:
    occ = occupancy.copy()
    for b in boxes:
        occ[b.x0 : b.x1, b.y0 : b.y1, b.z0 : b.z1] = False
    return occ


@dataclass
class BoxDecomposer(MeshDecomposer):
    """
    Decompose a mesh into boxes using voxelization.
    This is very efficient and works well for blocky furniture.
    This works poorly for non-blocky furniture.

    A board in this context is something like a board (the supporting surfaces that hold books) in a bookshelf.

    The algorithm works by first voxelizing the mesh and removing thin voxel cracks.
    Next, it detects planar boards and then for each axis (X, Y, Z):

        Extract thin slabs (1–N voxels thick)

        Deduplicate boards

        Remove overlapping duplicates, preferring thinner boards.

        Merge leftovers


    The results are that large planar structures (shelves, walls) become clean single boxes,
    while smaller details are handled separately.
    """

    voxel_size: float = 0.02
    """
    Voxel size in mesh units.
    """

    fill_thin_holes: bool = True
    """
    Rather to filling 1-voxel cracks/voids or not
    """

    max_board_thickness: int = 2
    """
    Maximum board thickness in voxels.
    If a board is bigger than this in the Z direction, it will be clipped to this thickness.
    """

    min_span_voxel: int = 3
    """
    Threshold for keeping boards.
    If a board has less than this voxels in the XY plane, it is removed.
    You can control the bumpiness in the XY plane using this parameter.
    """

    min_fill_ratio: float = 0.75
    """
    TODO
    """

    overlap_threshold: float = 0.8
    """
    Overlap threshold at which two boards are merged into one. 
    """

    def apply_to_mesh(self, mesh: Mesh) -> List[Box]:
        trimesh_mesh = mesh.mesh
        voxelized = trimesh_mesh.voxelized(pitch=self.voxel_size).fill()

        occupancy = voxelized.matrix.astype(bool)
        origin = np.asarray(voxelized.translation, dtype=float)

        occupancy = clean_occupancy(
            occupancy,
            fill_thin_holes=self.fill_thin_holes,
        )

        board_candidates = detect_planar_boards(
            occupancy=occupancy,
            max_thickness_vox=self.max_board_thickness,
            min_span_vox=self.min_span_voxel,
            min_fill_ratio=self.min_fill_ratio,
        )
        board_index_boxes = deduplicate_index_boxes(
            board_candidates,
            overlap_ratio_threshold=self.overlap_threshold,
        )
        board_boxes = [
            index_box_to_box(b, self.voxel_size, origin) for b in board_index_boxes
        ]

        remainder_occupancy = subtract_index_boxes_from_occupancy(
            occupancy, board_index_boxes
        )
        leftover_boxes = greedy_merge_boxes(
            remainder_occupancy, self.voxel_size, origin
        )

        all_boxes = board_boxes + leftover_boxes

        return [
            Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    *box.position, reference_frame=mesh.origin.reference_frame
                ),
                scale=Scale(*box.size),
            )
            for box in all_boxes
        ]
