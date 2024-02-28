import numpy as np
from numpy.typing import NDArray

from toothwear.teeth import DentalMesh


class TestGapException(Exception):
    pass


def is_next_inside(
    test: DentalMesh,
    triangle_idx: int,
    point: NDArray[np.float64],
) -> bool:
    triangle = test.triangles[triangle_idx]
    triangle_vertices = test.vertices[triangle]

    uvw = barycentric_coords(point, triangle_vertices)

    return np.all(uvw > 0)


def is_next_vertex(
    test: DentalMesh,
    plane_eq: NDArray[np.float64],
    test_edge_vector: NDArray[np.float64],
    triangle_idx: int,
    point: NDArray[np.float64],
) -> bool:
    triangle = test.triangles[triangle_idx]
    triangle_vertices = test.vertices[triangle]
    
    tri_intersections = plane_triangle_edges_intersections(
        plane_eq, triangle_vertices,
    )
    uvws = barycentric_coords(tri_intersections, triangle_vertices)
    uvws = remove_inverted_intersections(
        uvws, triangle_vertices, point, test_edge_vector,
    )

    return np.any(uvws == 1)


def next_triangle_idx(
    current_triangle_idx: int,
    next_edge_idx: NDArray[np.int64],
    test_edges: NDArray[np.int64],
) -> int:
    num_triangles = int(test_edges.shape[0] / 3)
    next_edge = test_edges[current_triangle_idx + next_edge_idx * num_triangles]
    edge_mask = np.all(test_edges == next_edge, axis=-1)
    triangle_idxs = np.nonzero(edge_mask)[0] % num_triangles
    other_triangle_idxs = triangle_idxs[triangle_idxs != current_triangle_idx]

    return other_triangle_idxs[0] if other_triangle_idxs.shape[0] > 0 else -1


def remove_inverted_intersections(
    uvws: NDArray[np.float64],
    triangle_vertices: NDArray[np.float64],
    intersection: NDArray[np.float64],
    test_edge_vector: NDArray[np.float64],
    eps: float=1e-4,
):
    projections = uvws @ triangle_vertices
    norm_mask = ~np.any(np.isnan(uvws), axis=-1)

    inter_vectors = projections - intersection
    directions = np.full(3, fill_value=-np.inf)
    directions[norm_mask] = test_edge_vector @ inter_vectors[norm_mask].T

    uvws[np.all(uvws < 1, axis=-1) & (directions < -eps)] = np.nan
    
    return uvws


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
    eps: float=1e-4,
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
