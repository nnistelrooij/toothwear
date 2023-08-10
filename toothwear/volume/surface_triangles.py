from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from toothwear.teeth import DentalMesh


def is_degenerate_triangle(
    vertices: NDArray[np.float32],
    eps: float=1e-2,
) -> bool:    
    a = np.linalg.norm(vertices[1] - vertices[0])
    b = np.linalg.norm(vertices[2] - vertices[1])
    c = np.linalg.norm(vertices[2] - vertices[0])
    a, b, c = np.sort([a, b, c]) / np.max([a, b, c])

    return a + b - c < eps


def determine_triangles_from_surface(
    surface: DentalMesh,
    extra: DentalMesh,
    boundary: NDArray[np.int64],
    intersections: List[NDArray[np.float64]],
    vertex_idxs_list: List[NDArray[np.int64]],
    init_idx: int,
    eps: float=1e-3,
) -> DentalMesh:
    for i, vertex_idxs in enumerate(vertex_idxs_list):
        vertex_idxs_list[i][vertex_idxs < init_idx] += surface.num_vertices

    triangles = np.ones((0, 3), dtype=int)
    max_vertex_idx = init_idx - 1
    for edge, inters, vertex_idxs in zip(boundary, intersections, vertex_idxs_list):
        start_idx, end_idx = edge
        edge_is_degenerate = False

        for k, (inter, vertex_idx) in enumerate(zip(inters[:-1], vertex_idxs[:-1])):
            if vertex_idx == vertex_idxs[k + 1]:
                continue

            if vertex_idx > max_vertex_idx:
                if vertex_idx > max_vertex_idx + 1:
                    l = 3
                extra.vertices = np.concatenate((extra.vertices, [inter]))
                extra.normals = np.concatenate((extra.normals, [surface.normals[start_idx]]))
                max_vertex_idx = vertex_idx

            vertices = np.stack((
                surface.vertices[start_idx],
                inters[k],
                inters[k + 1],
            ))
            triangle_is_degenerate = (
                not edge_is_degenerate
                and is_degenerate_triangle(vertices)
            )

            if triangle_is_degenerate:
                triangle = [end_idx, start_idx, vertex_idx]
                triangles = np.concatenate((triangles, [triangle]))

            edge_is_degenerate |= triangle_is_degenerate
            if edge_is_degenerate:
                triangle = [vertex_idx, vertex_idxs[k + 1], end_idx]
            else:
                triangle = [start_idx, vertex_idx, vertex_idxs[k + 1]]
            triangles = np.concatenate((triangles, [triangle]))

        if not edge_is_degenerate:
            triangle = [end_idx, start_idx, vertex_idxs[k + 1]]
            triangles = np.concatenate((triangles, [triangle]))
    
    # special case for last triangle
    assert end_idx == boundary[0, 0]
    triangles[triangles == vertex_idxs_list[-1][-1]] = vertex_idxs_list[0][0]

    extra.triangles = np.concatenate((extra.triangles, triangles))

    return extra
