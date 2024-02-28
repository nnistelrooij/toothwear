import numpy as np
from numpy.typing import NDArray

from toothwear.teeth import DentalMesh
from toothwear.volume.intersections.utils import (
    is_next_inside,
    is_next_vertex,
    barycentric_coords,
    next_triangle_idx,
    remove_inverted_intersections,
    plane_triangle_edges_intersections,
    TestGapException,
)
from toothwear.volume.intersections.from_inside_intersections import determine_inside_vertex_intersections


def determine_edge_test_intersections(
    test: DentalMesh,
    inters: NDArray[np.float64],
    vertex_idxs: NDArray[np.int64],
    test_triangle_idxs: NDArray[np.int64],
    current_triangle_idx: int,
    plane_eq: NDArray[np.float64],
    test_edge_vector: NDArray[np.float64],
    test_edges: NDArray[np.int64],
    test_edge_vertices: NDArray[np.int64],
    test_edge_triangle_idxs: NDArray[np.int64],
    init_idx: int,
):
    if is_next_inside(test, current_triangle_idx, test_edge_vertices[1]):
        # edge to inside
        inters = np.concatenate((inters, [test_edge_vertices[1]]))
        vertex_idxs = np.concatenate((vertex_idxs, [init_idx]))
        test_triangle_idxs = np.concatenate((test_triangle_idxs, [current_triangle_idx]))
            
        return inters, vertex_idxs, test_triangle_idxs, False, False, -1, True
    
    if is_next_vertex(test, plane_eq, test_edge_vector, current_triangle_idx, inters[-1]):
        # edge to vertex
        return determine_inside_vertex_intersections(
            test, inters, vertex_idxs, test_triangle_idxs, plane_eq,
            test_edge_vector, test_triangle_idxs, test_edges, current_triangle_idx,
        )
    
    # edge to edge
    triangle = test.triangles[current_triangle_idx]
    triangle_vertices = test.vertices[triangle]
    
    tri_intersections = plane_triangle_edges_intersections(
        plane_eq, triangle_vertices,
    )
    uvws = barycentric_coords(tri_intersections, triangle_vertices)
    uvws = remove_inverted_intersections(
        uvws, triangle_vertices, inters[-1], test_edge_vector,
    )

    # next is on edge
    # assert np.any(uvws == 0, axis=-1).sum() == 2
    projections = uvws @ triangle_vertices
    norm_mask = ~np.any(np.isnan(uvws), axis=-1)

    inter_vectors = projections - inters[-1]
    directions = np.full(3, fill_value=-np.inf)
    directions[norm_mask] = test_edge_vector @ inter_vectors[norm_mask].T
    # assert (directions > 1e-5).sum() == 1

    edge_idx = directions.argmax()
    current_vertex = projections[edge_idx]
    current_triangle_idx = next_triangle_idx(
        current_triangle_idx, edge_idx, test_edges,
    )

    if current_triangle_idx == -1:
        raise TestGapException

    # add stats of current point
    inters = np.concatenate((inters, [current_vertex]))
    init_idx += np.linalg.norm(inters[-1] - inters[-2]) >= 1e-3
    vertex_idxs = np.concatenate((vertex_idxs, [init_idx]))
    test_triangle_idxs = np.concatenate((test_triangle_idxs, [current_triangle_idx]))

    return inters, vertex_idxs, test_triangle_idxs, False, True, -1, current_triangle_idx

