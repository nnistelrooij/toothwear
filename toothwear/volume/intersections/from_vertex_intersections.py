import numpy as np
from numpy.typing import NDArray

from toothwear.teeth import DentalMesh
from toothwear.volume.intersections.utils import (
    barycentric_coords,
    next_triangle_idx,
    remove_inverted_intersections,
    plane_triangle_edges_intersections,
    TestGapException,
)

def determine_vertex_vertex_intersections(
    test: DentalMesh,
    current_vertex_idx: int,
    current_triangle_idx: int,
    test_edges: NDArray[np.int64],
    vertex_idx: int,
    triangle: NDArray[np.int64],
):
    triangle_edges = test_edges[current_triangle_idx::test.num_triangles]
    edge_idx = np.nonzero(
        np.any(triangle_edges == vertex_idx, axis=-1)
        &
        np.all(triangle_edges != current_vertex_idx, axis=-1)
    )[0][0]
    current_triangle_idx = next_triangle_idx(
        current_triangle_idx, edge_idx, test_edges,
    )
    if current_triangle_idx == -1:
        raise TestGapException


    current_vertex_idx = triangle[
        (triangle != vertex_idx)
        &
        (triangle != current_vertex_idx)
    ][0]
    
    return current_vertex_idx, current_triangle_idx
    


def determine_vertex_test_intersections(
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
    current_vertex_idx: int,
    init_idx: int,
):
    if np.linalg.norm(inters[-1] - test_edge_vertices[1]) < 1e-3:
        # vertex of current triangle to same vertex of final triangle
        test_triangle_idxs[-1] = test_edge_triangle_idxs[1]
        return inters, vertex_idxs, test_triangle_idxs, True, False, -1, test_edge_triangle_idxs[1]
    
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
    vertex_idxs_ = np.argsort(vertex_uvws.flatten()) % 3
    vertex_idxs_ = vertex_idxs_[-(uvws == 1).sum():]
    unique, counts = np.unique(vertex_idxs_, return_counts=True)
    if vertex_uvws.shape[0] < 3:  # and np.any(counts == 2):
        # vertex of current triangle to same vertex of next triangle
        if np.any(counts == 2):
            vertex_idx = triangle[unique[counts == 2][0]]
        else:
            uvws = barycentric_coords(tri_intersections, triangle_vertices)
            vertex_uvws = uvws[~np.any(np.isnan(uvws), axis=-1)]
            vertex_idx = triangle[vertex_uvws.argmax() % 3]
            
        current_vertex_idx, current_triangle_idx = determine_vertex_vertex_intersections(
            test, current_vertex_idx, current_triangle_idx,
            test_edges, vertex_idx, triangle,
        )

        test_triangle_idxs[-1] = current_triangle_idx

        return inters, vertex_idxs, test_triangle_idxs, True, False, current_vertex_idx, current_triangle_idx

    if np.all(np.any(uvws == 1, axis=-1)):
        # vertex of current triangle to connected edge of current triangle
        current_triangle_vertex_idx = unique[counts == 2][0]
        next_triangle_vertex_idx = unique[counts == 1][0]
        edge = np.sort(np.array([current_triangle_vertex_idx, next_triangle_vertex_idx]), axis=-1)
        edge_idx = edge[1] - ((edge[1] - edge[0]) == 1)
        opposite_triangle_idx = next_triangle_idx(
            current_triangle_idx, edge_idx, test_edges,
        )

        if opposite_triangle_idx == test_edge_triangle_idxs[1]:
            # go to opposite triangle to finish surface edge
            test_triangle_idxs[-1] = opposite_triangle_idx
            return inters, vertex_idxs, test_triangle_idxs, True, False, -1, opposite_triangle_idx
        else:
            inters = np.concatenate((inters, [triangle_vertices[next_triangle_vertex_idx]]))
            vertex_idxs = np.concatenate((vertex_idxs, [triangle[next_triangle_vertex_idx]]))
            test_triangle_idxs = np.concatenate((test_triangle_idxs, [current_triangle_idx]))

            return inters, vertex_idxs, test_triangle_idxs, True, False, -1, current_triangle_idx

    
    # vertex of current triangle to opposite edge of current triangle
    test_triangle_idxs[-1] = current_triangle_idx
    edge_idx = np.nonzero(np.all(uvws != 1, axis=-1))[0][0]
    current_triangle_idx = next_triangle_idx(
        current_triangle_idx, edge_idx, test_edges,
    )

    inters = np.concatenate((inters, [uvws[edge_idx] @ triangle_vertices]))
    vertex_idxs = np.concatenate((vertex_idxs, [init_idx + 1]))
    test_triangle_idxs = np.concatenate((test_triangle_idxs, [current_triangle_idx]))

    return inters, vertex_idxs, test_triangle_idxs, False, True, -1, current_triangle_idx        
