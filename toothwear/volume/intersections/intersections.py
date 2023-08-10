from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from toothwear.teeth import DentalMesh
from toothwear.volume.intersections.from_inside_intersections import determine_inside_test_intersections
from toothwear.volume.intersections.from_vertex_intersections import determine_vertex_test_intersections
from toothwear.volume.intersections.from_edge_intersections import determine_edge_test_intersections
from toothwear.volume.intersections.utils import (
    barycentric_coords,
    next_triangle_idx,
    edge_plane_equation,
)


def project_intersections_to_vertex_or_edge(
    intersections: NDArray[np.float64],
    test_triangle_idxs: NDArray[np.int64],
    vertex_idxs: NDArray[np.int64],
    test: DentalMesh,
) -> Tuple[bool, bool]:
    on_vertex, on_edge = False, False
    for i in range(2):
        triangle = test.triangles[test_triangle_idxs[i]]
        triangle_vertices = test.vertices[triangle]
        uvw = barycentric_coords(intersections[i], triangle_vertices)

        if np.any(uvw == 0):
            # current intersection is on edge
            intersections[i] = uvw @ triangle_vertices

            on_edge = i == 0
            if np.any(uvw == 1) and (i == 0):
                # current intersection is on vertex
                vertex_idxs[i] = triangle[uvw.argmax()]

                on_vertex = True

    return on_vertex, on_edge


def add_opposite_edge(
    test: DentalMesh,
    inters: NDArray[np.float64],
    vertex_idxs: NDArray[np.int64],
    test_triangle_idxs: NDArray[np.int64],
    test_edges: NDArray[np.int64],
):
    triangle = test.triangles[test_triangle_idxs[-1]]
    triangle_vertices = test.vertices[triangle]

    uvw = barycentric_coords(inters[-1], triangle_vertices)

    if np.all(uvw != 0) or np.any(uvw == 1):
        return inters, vertex_idxs, test_triangle_idxs
    
    edge_idx = (uvw.argmin() + 1) % 3
    triangle_idx = next_triangle_idx(test_triangle_idxs[-1], edge_idx, test_edges)

    if (
        (test_triangle_idxs.shape[0] >= 2 and
        triangle_idx == test_triangle_idxs[-2])
        or np.any(test_triangle_idxs == triangle_idx)
    ):
        return inters, vertex_idxs, test_triangle_idxs


    inters = np.concatenate((inters, [inters[-1]]))
    vertex_idxs = np.concatenate((vertex_idxs, [vertex_idxs[-1]]))
    test_triangle_idxs = np.concatenate((test_triangle_idxs, [triangle_idx]))
    
    return inters, vertex_idxs, test_triangle_idxs
    

def add_final_point(
    test: DentalMesh,
    inters: NDArray[np.float64],
    vertex_idxs: NDArray[np.int64],
    triangle_idx: int,
    test_edge_vertices: NDArray[np.float64],
    init_idx: int,
):   
    triangle = test.triangles[triangle_idx]
    triangle_vertices = test.vertices[triangle]

    uvw = barycentric_coords(test_edge_vertices[1], triangle_vertices)

    if np.any(uvw == 1):
        vertex_idx = triangle[uvw.argmax()]
        vertex_idxs = np.concatenate((vertex_idxs, [vertex_idx]))
    else:
        init_idx += np.linalg.norm(inters[-1] - test_edge_vertices[1]) >= 1e-3
        vertex_idxs = np.concatenate((vertex_idxs, [init_idx]))
    
    inters = np.concatenate((inters, [test_edge_vertices[1]]))

    return inters, vertex_idxs  


def determine_surface_edge_test_intersections(
    surface_edge_vertices: NDArray[np.float64],
    test_edge_vertices: NDArray[np.float64],
    test_edge_triangle_idxs: NDArray[np.int64],
    test: DentalMesh,
    init_idx: int,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    vertex_idxs = np.array([init_idx])
    current_on_vertex, current_on_edge = project_intersections_to_vertex_or_edge(
        test_edge_vertices, test_edge_triangle_idxs, vertex_idxs, test,
    )
    
    # add stats of first test point
    inters = test_edge_vertices[:1]
    test_triangle_idxs = test_edge_triangle_idxs[:1]

    # determine plane going through edge
    test_edge_vector = test_edge_vertices[1] - test_edge_vertices[0]
    plane_eq = edge_plane_equation(
        surface_edge_vertices, test_edge_vertices,
    )
    test_edges = np.sort(test.edges, axis=-1)

    current_triangle_idx = test_triangle_idxs[0]
    current_vertex_idx = -1
    while current_triangle_idx != test_edge_triangle_idxs[1]:
        if current_on_vertex:
            results = determine_vertex_test_intersections(
                test, inters, vertex_idxs, test_triangle_idxs,
                current_triangle_idx, plane_eq, test_edge_vector,
                test_edges, test_edge_vertices,
                test_edge_triangle_idxs, current_vertex_idx, init_idx,
            )
        elif current_on_edge:
            results = determine_edge_test_intersections(
                test, inters, vertex_idxs, test_triangle_idxs,
                current_triangle_idx, plane_eq, test_edge_vector,
                test_edges, test_edge_vertices,
                test_edge_triangle_idxs, init_idx,
            )
        else:
            results = determine_inside_test_intersections(
                test, inters, vertex_idxs, test_triangle_idxs,
                current_triangle_idx, plane_eq, test_edge_vector,
                test_edges, test_edge_vertices,
                test_edge_triangle_idxs, init_idx,
            )

        inters, vertex_idxs, test_triangle_idxs = results[:3]
        current_on_vertex, current_on_edge, current_vertex_idx, current_triangle_idx = results[3:]

        inters, vertex_idxs, test_triangle_idxs = add_opposite_edge(
            test, inters, vertex_idxs, test_triangle_idxs,
            test_edges,
        )

        init_idx = max(init_idx, vertex_idxs.max())


    inters, vertex_idxs = add_final_point(
        test, inters, vertex_idxs, current_triangle_idx,
        test_edge_vertices, init_idx,
    )
    inters, vertex_idxs, test_triangle_idxs = add_opposite_edge(
        test, inters, vertex_idxs, test_triangle_idxs,
        test_edges,
    )

    return inters, vertex_idxs, test_triangle_idxs


def determine_boundary_test_intersections(
    surface : DentalMesh,
    test: DentalMesh,
    edges: NDArray[np.int64],
    init_idx: int,
) -> Tuple[
    List[NDArray[np.float64]],
    List[NDArray[np.int64]],
]:    
    closest_points = surface.closest_points(test)

    edge_inters, edge_vertex_idxs, edge_test_triangle_idxs = [], [], []
    for edge in edges:
        if np.any(edge == 83):
            k = 3

        surface_edge_vertices = surface.vertices[edge]
        test_edge_vertices = closest_points['points'][edge]
        test_edge_triangle_idxs = closest_points['primitive_ids'][edge]

        inters, vertex_idxs, test_triangle_idxs = determine_surface_edge_test_intersections(
            surface_edge_vertices,
            test_edge_vertices,
            test_edge_triangle_idxs,
            test,
            init_idx,
        )

        if np.any(test_triangle_idxs == 5487):
            k = 3

        edge_inters.append(inters)
        edge_vertex_idxs.append(vertex_idxs)
        edge_test_triangle_idxs.append(test_triangle_idxs)
        init_idx = max(init_idx, vertex_idxs.max())


    edge_vertex_idxs[-1][edge_vertex_idxs[-1] == edge_vertex_idxs[-1][-1]] = edge_vertex_idxs[0][0]
            
    return edge_inters, edge_vertex_idxs, edge_test_triangle_idxs
