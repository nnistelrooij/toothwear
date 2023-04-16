from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from toothwear.teeth import DentalMesh


def edge_plane_equation(
    surface_edge_vertices: NDArray[np.float64],
    test_edge_vertices: NDArray[np.float64],
) -> NDArray[np.float64]:
    # determine vectors in plane
    vector1 = surface_edge_vertices[0] - test_edge_vertices[0]
    vector2 = test_edge_vertices[1] - test_edge_vertices[0]

    # determine plane normal
    plane_normal = np.cross(vector1, vector2)
    plane_normal /= np.linalg.norm(plane_normal)

    # determine plane constant
    plane_d = surface_edge_vertices[0] @ plane_normal

    # determine scalars of plane equation
    plane_eq = np.concatenate((plane_normal, [-plane_d]))

    return plane_eq


def plane_triangle_edges_intersections(
    inters: NDArray[np.float64],
    plane_eq: NDArray[np.float64],
    triangle_vertices: NDArray[np.float64],
    eps: float=1e-5,
) -> NDArray[np.float64]:
    # determine intersections of edge plane with current triangle
    edge_vertices = np.stack((
        triangle_vertices[:2], triangle_vertices[1:], triangle_vertices[[2, 0]],
    ))

    u = edge_vertices[:, 1] - edge_vertices[:, 0]
    dot = plane_eq[:3] @ u.T

    p_co = -plane_eq[:3] * plane_eq[3]
    w = edge_vertices[:, 0] - p_co
    fac = -plane_eq[:3] @ w.T / dot
    intersections = edge_vertices[:, 0] + (u * fac[:, np.newaxis])

    outside = (
        (intersections < edge_vertices.min(axis=1) - eps)
        | (edge_vertices.max(axis=1) + eps < intersections)
    )
    outside = np.any(outside, axis=-1)
    intersections[outside] = np.inf

    return intersections


def barycentric_coords(
    p: NDArray[np.float32],
    triangle: NDArray[np.float32],
    eps: float=1e-5,
) -> NDArray[np.float32]:
    # https://ceng2.ktu.edu.tr/~cakir/files/grafikler/Texture_Mapping.pdf
    a, b, c = triangle
    v0, v1, v2 = b - a, c - a, p - a
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0
    d21 = v2 @ v1
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    # always return NumPy array
    if isinstance(u, np.ndarray):
        uv = np.column_stack((u, v))
    else:
        uv = np.array([u, v])

    # numerical stability
    uv[np.abs(uv) < eps] = 0
    uv[np.abs(uv) > 1 - eps] = 1
    w = 1 - uv[..., 0] - uv[..., 1]
    uv[..., 0] += np.where(np.abs(w) < eps, w, 0)
    w = 1 - uv[..., 0] - uv[..., 1]
    uvw = np.stack((uv[..., 0], uv[..., 1], w), axis=-1)
    uvw[np.abs(uvw) < eps] = 0

    return uvw


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
            if np.any(uvw == 1):
                # current intersection is on vertex
                vertex_idxs[i] = triangle[uvw.argmax()]

                on_vertex |= i == 0                

            on_edge |= i == 0

    return on_vertex, on_edge


def next_triangle_idx(
    current_triangle_idx: int,
    next_edge_idx: NDArray[np.int64],
    test_edges: NDArray[np.int64],
) -> int:
    num_triangles = int(test_edges.shape[0] / 3)
    next_edge = test_edges[current_triangle_idx + next_edge_idx * num_triangles]
    edge_mask = np.all(test_edges == next_edge, axis=-1)
    triangle_idxs = np.nonzero(edge_mask)[0] % num_triangles
    next_tri_idx =  triangle_idxs[triangle_idxs != current_triangle_idx][0]

    return next_tri_idx


def determine_edge_test_intersections(
    surface_edge_vertices: NDArray[np.float64],
    test_edge_vertices: NDArray[np.float64],
    test_edge_triangle_idxs: NDArray[np.int64],
    continued_triangle_idx: int,
    test: DentalMesh,
    init_idx: int,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    int,
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
    start, stop = True, test_edge_triangle_idxs[0] == test_edge_triangle_idxs[1]
    current_vertex_idx = -1
    while True:
        triangle = test.triangles[current_triangle_idx]
        triangle_vertices = test.vertices[triangle]
        triangle_edges = test_edges[current_triangle_idx::test.num_triangles]
        
        tri_intersections = plane_triangle_edges_intersections(
            inters, plane_eq, triangle_vertices,
        )
        uvws = barycentric_coords(tri_intersections, triangle_vertices)

        if np.any(uvws == 1):
            # next is on vertex
            vertex_idx = triangle[uvws.argmax()]
            if not current_on_vertex:
                inters = np.concatenate((inters, [triangle_vertices[uvws.argmax()]]))
                vertex_idxs = np.concatenate((vertex_idxs, [vertex_idx]))
                test_triangle_idxs = np.concatenate((test_triangle_idxs, current_triangle_idx))
                current_on_vertex = True

                continue

            edge_idx = np.nonzero(np.all(
                (triangle_edges == vertex_idx)
                | (triangle_edges != current_vertex_idx),
                axis=-1,
            ))[0][0]
            current_triangle_idx = next_triangle_idx(
                current_triangle_idx, edge_idx, test_edges,
            )
            current_vertex_idx = triangle[(
                triangle != vertex_idx) & (triangle != current_vertex_idx)
            ][0]
            continue

        # next is on edge
        assert np.any(uvws == 0, axis=-1).sum() == 2
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
        current_on_vertex = False
        current_vertex_idx = -1

        if current_triangle_idx in [471]:
            k = 3

        # if (
        #     start and
        #     continued_triangle_idx >= 0 and
        #     current_triangle_idx != continued_triangle_idx and
        #     test_edge_triangle_idxs[0] != continued_triangle_idx and
        #     current_on_edge
        # ):
        #     # add triangle if only one point is involved
        #     inters = np.concatenate(([inters[0]], inters))
        #     vertex_idxs = np.concatenate(([vertex_idxs[0]], vertex_idxs))
        #     test_triangle_idxs = np.concatenate(([continued_triangle_idx], test_triangle_idxs))


        if stop:
            # add stats of last point
            inters = np.concatenate((inters, [test_edge_vertices[1]]))
            
            init_idx += np.linalg.norm(inters[-1] - inters[-2]) >= 1e-5
            vertex_idxs = np.concatenate((vertex_idxs, [init_idx]))

            # if np.linalg.norm(inters[-1] - inters[-2]) < 1e-5:
            #     continued_triangle_idx = test_triangle_idxs[-1]
            # else:
            #     vertex_idxs = np.concatenate((vertex_idxs, [init_idx]))
            #     continued_triangle_idx = current_triangle_idx

            return inters, vertex_idxs, test_triangle_idxs, current_triangle_idx
        
        
        inters = np.concatenate((inters, [current_vertex]))
        init_idx += np.linalg.norm(inters[-1] - inters[-2]) >= 1e-5
        vertex_idxs = np.concatenate((vertex_idxs, [init_idx]))
        test_triangle_idxs = np.concatenate((test_triangle_idxs, [current_triangle_idx]))

        stop = current_triangle_idx == test_edge_triangle_idxs[1]
        start = False        


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
    continued_triangle_idx = -1
    for edge in edges:
        surface_edge_vertices = surface.vertices[edge]
        test_edge_vertices = closest_points['points'][edge]
        test_edge_triangle_idxs = closest_points['primitive_ids'][edge]

        results = determine_edge_test_intersections(
            surface_edge_vertices,
            test_edge_vertices,
            test_edge_triangle_idxs,
            continued_triangle_idx,
            test,
            init_idx,
        )
        inters, vertex_idxs, test_triangle_idxs, continued_triangle_idx = results

        edge_inters.append(inters)
        edge_vertex_idxs.append(vertex_idxs)
        edge_test_triangle_idxs.append(test_triangle_idxs)
        init_idx = max(init_idx, vertex_idxs.max())
            
    return edge_inters, edge_vertex_idxs, edge_test_triangle_idxs
