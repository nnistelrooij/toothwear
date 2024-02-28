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


def determine_inside_vertex_intersections(
    test: DentalMesh,
    inters: NDArray[np.float64],
    vertex_idxs: NDArray[np.int64],
    test_triangle_idxs: NDArray[np.int64],
    plane_eq: NDArray[np.float64],
    test_edge_vector: NDArray[np.float64],
    test_edge_triangle_idxs: NDArray[np.int64],
    test_edges: NDArray[np.int64],
    current_triangle_idx: int,
):
    triangle = test.triangles[current_triangle_idx]
    triangle_vertices = test.vertices[triangle]
    
    tri_intersections = plane_triangle_edges_intersections(
        plane_eq, triangle_vertices,
    )
    uvws = barycentric_coords(tri_intersections, triangle_vertices)
    uvws = remove_inverted_intersections(
        uvws, triangle_vertices, inters[-1], test_edge_vector,
    )

    vertex_uvws = uvws[~np.any(np.isnan(uvws), axis=-1)]
    if vertex_uvws.shape[0] > 1:
        vertex_idxs_ = np.argsort(vertex_uvws.flatten()) % 3
        vertex_idxs_ = vertex_idxs_[-vertex_uvws.shape[0]:]
        unique, counts = np.unique(vertex_idxs_, return_counts=True)

        # edge to opposite vertex
        if np.any(counts == 2):
            vertex_idx = triangle[unique[counts == 2][0]]
        else:
            vertex_idx = triangle[vertex_uvws[0].argmax()]
    else:
        vertex_idx = triangle[vertex_uvws[0].argmax()]
    inters = np.concatenate((inters, [test.vertices[vertex_idx]]))
    vertex_idxs = np.concatenate((vertex_idxs, [vertex_idx]))
    test_triangle_idxs = np.concatenate((test_triangle_idxs, [current_triangle_idx]))
    
    return inters, vertex_idxs, test_triangle_idxs, True, False, -1, current_triangle_idx


def determine_inside_edge_intersections(
    test: DentalMesh,
    inters: NDArray[np.float64],
    vertex_idxs: NDArray[np.int64],
    test_triangle_idxs: NDArray[np.int64],
    current_triangle_idx: int,
    plane_eq: NDArray[np.float64],
    test_edge_vector: NDArray[np.float64],
    test_edges: NDArray[np.int64],
    init_idx,
):
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

    


def determine_inside_test_intersections(
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
        # inside to inside
        inters = np.concatenate((inters, [test_edge_vertices[1]]))
        vertex_idxs = np.concatenate((vertex_idxs, [init_idx]))
        test_triangle_idxs = np.concatenate((test_triangle_idxs, [current_triangle_idx]))
            
        return inters, vertex_idxs, test_triangle_idxs, False, False, -1, True
    
    if is_next_vertex(test, plane_eq, test_edge_vector, current_triangle_idx, inters[-1]):
        # inside to vertex
        return determine_inside_vertex_intersections(
            test, inters, vertex_idxs, test_triangle_idxs, plane_eq,
            test_edge_vector, test_triangle_idxs, test_edges, current_triangle_idx,
        )
    
    return determine_inside_edge_intersections(
        test, inters, vertex_idxs, test_triangle_idxs,
        current_triangle_idx, plane_eq, test_edge_vector, test_edges, init_idx,
    )