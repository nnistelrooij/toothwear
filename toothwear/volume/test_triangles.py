from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from triangle import triangulate

from toothwear.teeth import DentalMesh
from toothwear.volume.intersections.intersections import barycentric_coords


def remove_vertices(
    uvws: NDArray[np.float64],
    internal_edges: NDArray[np.int64],
) -> Tuple[
    NDArray[np.bool_],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    is_vertex = np.any(uvws == 1, axis=-1)

    if is_vertex.sum() > 3:
        k = 3

    vertex_map = np.cumsum(~is_vertex) + 2
    vertex_map[is_vertex] = uvws[is_vertex].argmax(axis=-1)
    internal_edges = internal_edges + 3
    internal_edges = vertex_map[internal_edges]

    uvws = np.concatenate((uvws[:3], uvws[~is_vertex]))

    return ~is_vertex[3:], uvws, internal_edges
    

def delaunay_triangulation(
    triangle_vertices: NDArray[np.float64],
    points: NDArray[np.float64],
    internal_edges: NDArray[np.int64],
    verbose: bool=False,
) -> Tuple[
    NDArray[np.int64],
    NDArray[np.bool_],
]:
    points = np.concatenate((triangle_vertices, points))
    uvws = barycentric_coords(points, triangle_vertices)
    keep_mask, uvws, internal_edges = remove_vertices(uvws, internal_edges)
    
    # determine index of triangle edge of each edge vertex
    is_border = np.any(uvws[3:] == 0, axis=-1)
    border_idxs = np.full(is_border.shape[0], fill_value=-1)
    border_idxs[is_border] = uvws[3:][is_border].argmin(axis=-1)

    # determine edges on border of triangle
    border0 = (border_idxs == 0) + uvws[3:, 1]
    offset = border0.shape[0] - (border_idxs == 0).sum()
    border0 = border0.argsort()[offset:]
    border0 = np.concatenate(([2], 3 + border0, [1]))

    border1 = (border_idxs == 1) + uvws[3:, 0]
    offset = border1.shape[0] - (border_idxs == 1).sum()
    border1 = border1.argsort()[offset:]
    border1 = np.concatenate(([2], 3 + border1, [0]))

    border2 = (border_idxs == 2) + uvws[3:, 0]
    offset = border2.shape[0] - (border_idxs == 2).sum()
    border2 = border2.argsort()[offset:]
    border2 = np.concatenate(([1], 3 + border2, [0]))

    # remove degenerate and duplicate edges
    segments = np.concatenate((
        internal_edges,
        np.column_stack((border0[:-1], border0[1:])),
        np.column_stack((border1[:-1], border1[1:])),
        np.column_stack((border2[:-1], border2[1:])),
    ))
    segments = segments[segments[:, 0] != segments[:, 1]]
    segments = np.unique(segments, axis=0)

    # constrained delaunay triangulation
    tri_dict = {
        'vertices': uvws[:, :2],
        'segments': segments,
    }
    tri = triangulate(tri_dict, 'p')
    triangles = tri['triangles'].astype(np.int64)

    if verbose:
        plt.triplot(uvws[:, 0], uvws[:, 1], triangles)
        plt.title(f'Number of triangles {triangles.shape[0]}')
    
    return triangles, keep_mask


def determine_triangles_from_test(
    surface: DentalMesh,
    test: DentalMesh,
    extra: DentalMesh,
    triangle_edges_map: Dict[
        int,
        Tuple[NDArray[np.int64], NDArray[np.float32]],
    ],
) -> DentalMesh:
    test_triangle_idxs = np.array(list(triangle_edges_map.keys()))
    for test_triangle_idx in test_triangle_idxs:
        vertex_idxs, intersections = triangle_edges_map[test_triangle_idx]

        unique, index, inverse = np.unique(
            vertex_idxs, return_index=True, return_inverse=True,
        )

        if test_triangle_idx == 5487:
            k = 3

        # determine internal triangles of test triangle
        test_triangle = test.triangles[test_triangle_idx]
        triangles, keep_mask = delaunay_triangulation(
            test.vertices[test_triangle],
            intersections[index],
            inverse.reshape(-1, 2),
            # verbose=verbose,
        )
        
        # translate vertex indices to accomodate surface and test triangles
        triangles = triangles - 3
        triangles[triangles >= 0] = unique[keep_mask][triangles[triangles >= 0]]
        triangles[triangles < 0] = test_triangle[triangles[triangles < 0] + 3]
        triangles[triangles < test.num_vertices] += surface.num_vertices

        extra.triangles = np.concatenate((extra.triangles, triangles[:, [1, 0, 2]]))

    extra.triangles = np.unique(extra.triangles, axis=0)

    return extra
