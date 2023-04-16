from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from toothwear.teeth import DentalMesh


def update_boundaries(
    boundaries: List[NDArray[np.int64]],
    starts: NDArray[np.int64],
    stops: NDArray[np.int64],
    edge: NDArray[np.int64],
    vertex_idx: int,
) -> Tuple[
    List[NDArray[np.int64]],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    # add edge to boundary
    boundary1_idx = np.argmax(starts == edge[vertex_idx])
    boundaries[boundary1_idx] = np.concatenate((
        edge[np.newaxis][:vertex_idx],
        boundaries[boundary1_idx],
        edge[np.newaxis][:1 - vertex_idx],
    ))

    if boundaries[boundary1_idx][0, 0] == boundaries[boundary1_idx][-1, 1]:
        # boundary is complete
        starts[boundary1_idx] = -1
        stops[boundary1_idx] = -1
        return boundaries, starts, stops
    
    if not np.any(stops == edge[1 - vertex_idx]):
        # no boundaries can be combined
        starts[boundary1_idx] = edge[1 - vertex_idx]
        return boundaries, starts, stops
    
    # combine boundaries by connecting start and stop
    boundary2_idx = np.argmax(stops == edge[1 - vertex_idx])
    boundaries[boundary2_idx] = np.concatenate((
        boundaries[boundary2_idx] if vertex_idx == 1 else np.ones((0, 2), dtype=int),
        boundaries[boundary1_idx],
        boundaries[boundary2_idx] if vertex_idx == 0 else np.ones((0, 2), dtype=int),
    ))
    stops[boundary2_idx] = stops[boundary1_idx]

    boundaries = boundaries[:boundary1_idx] + boundaries[boundary1_idx + 1:]
    starts = np.delete(starts, boundary1_idx)
    stops = np.delete(stops, boundary1_idx)

    return boundaries, starts, stops


def identify_loops(
    edges: NDArray[np.int64],
) -> List[NDArray[np.int64]]:    
    tails, heads = np.ones((2, 0), dtype=int)
    boundaries = []
    for edge in edges:
        # add edge to tail of a boundary and combine boundaries
        if np.any(tails == edge[1]):
            boundaries, tails, heads = update_boundaries(
                boundaries, tails, heads, edge, 1,
            )
            continue

        # add edge to head of a boundary and combine boundaries
        if np.any(heads == edge[0]):
            boundaries, heads, tails = update_boundaries(
                boundaries, heads, tails, edge, 0,
            )
            continue

        # make new boundary
        boundaries.append(edge[np.newaxis])
        tails = np.concatenate((tails, [edge[0]]))
        heads = np.concatenate((heads, [edge[1]]))
        
    return boundaries


def combine_loops(
    boundaries: List[NDArray[np.int64]],
) -> List[NDArray[np.int64]]:
    keep_idxs = list(range(len(boundaries)))
    for i, edges1 in enumerate(boundaries):
        for j, edges2 in enumerate(boundaries[i + 1:], i + 1):
            mask = edges1[np.newaxis, :, 0] == edges2[:, 0, np.newaxis]
            if not np.any(mask):
                continue

            # reorder first boundary to have common point at tail and head
            edges1_idx = np.nonzero(np.any(mask, axis=0))[0][0]
            edges1 = np.concatenate((edges1[edges1_idx:], edges1[:edges1_idx]))

            # insert first boundary into second boundary
            edges2_idx = np.nonzero(np.any(mask, axis=1))[0][0]
            edges2 = np.concatenate((
                edges2[:edges2_idx], edges1, edges2[edges2_idx:],
            ))

            # save combined boundary
            boundaries[j] = edges2
            keep_idxs.remove(i)

    boundaries = [edges for i, edges in enumerate(boundaries) if i in keep_idxs]

    return boundaries


def duplicate_common_point(
    surface: DentalMesh,
    boundary: NDArray[np.int64],
    vertex_idx: int,
    dist: float=1e-3,
) -> None:        
    vertex = surface.vertices[vertex_idx]

    # get opposite edges from common point
    triangle_mask = np.any(surface.triangles == vertex_idx, axis=-1)
    triangle_idxs = np.nonzero(triangle_mask)[0]
    triangles = surface.triangles[triangle_idxs]
    edges = np.concatenate((
        triangles[:, :2], triangles[:, 1:], triangles[:, [2, 0]],
    ))
    opposite_edges = edges[~np.any(edges == vertex_idx, axis=-1)]
    
    # identify connected edges
    groups = identify_loops(opposite_edges)

    for i, group in enumerate(groups):
        # get difference to new position given group centroid
        vertex_idxs = np.unique(group)
        centroid = surface.vertices[vertex_idxs].mean(axis=0)
        direction = centroid - vertex
        diff = dist * direction / np.linalg.norm(direction)

        if i == 0:
            surface.vertices[vertex_idx] += diff
            continue

        # map triangles
        edges_mask = np.any(np.all(
            group[np.newaxis] == edges[:, np.newaxis],
        axis=-1), axis=1)
        edge_idxs = np.nonzero(edges_mask)[0]
        group_triangle_idxs = triangle_idxs[edge_idxs % triangle_mask.sum()]
        group_triangle_mask = np.zeros((surface.num_triangles, 3), dtype=bool)
        group_triangle_mask[group_triangle_idxs] = True
        vertex_mask = group_triangle_mask & (surface.triangles == vertex_idx)
        surface.triangles[vertex_mask] = surface.num_vertices
        
        # map boundary
        edges_mask = np.any(np.any(
            group[np.newaxis, np.newaxis] == boundary[:, np.newaxis, :, np.newaxis],
        axis=(-2, -1)), axis=1, keepdims=True)
        vertex_mask = edges_mask & (boundary == vertex_idx)
        boundary[vertex_mask] = surface.num_vertices

        # add new vertex
        surface.vertices = np.concatenate((surface.vertices, [vertex + diff]))
        surface.normals = np.concatenate((surface.normals, [surface.normals[vertex_idx]]))        
        surface.labels = np.concatenate((surface.labels, [surface.labels[vertex_idx]]))


def duplicate_common_points(
    surface: DentalMesh,
    boundaries: List[NDArray[np.int64]],
):
    for edges in boundaries:
        unique, counts = np.unique(edges, return_counts=True)
        if counts.max() == 2:
            continue

        for vertex_idx in unique[counts > 2]:
            duplicate_common_point(surface, edges, vertex_idx)


def compute_boundaries(
    surface: DentalMesh,
) -> List[NDArray[np.int64]]:
    boundaries = identify_loops(surface.border_edges())
    boundaries = combine_loops(boundaries)
    duplicate_common_points(surface, boundaries)

    return boundaries
