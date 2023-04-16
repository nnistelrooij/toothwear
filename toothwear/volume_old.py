from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
from numpy.typing import NDArray
import open3d
from scipy.spatial import Delaunay
from triangle import triangulate
from tqdm import tqdm

from toothwear.teeth import DentalMesh


def intersect(
    reference: DentalMesh,
    test: DentalMesh,
) -> Tuple[
    NDArray[np.float32],
    NDArray[np.float32],
]:
    closest_points = reference.closest_points(test)

    signs = np.sign(closest_points['signed_distances'])
    intersections = closest_points['points']

    return signs, intersections


def triangle_areas(
    triangles: NDArray[np.float32],
) -> NDArray[np.float32]:
    a = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=-1)
    b = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=-1)
    c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=-1)
    s = (a + b + c) / 2
    areas = np.sqrt(s * (s - a) * (s - b) * (s - c))

    return areas


def barycentric_coords(
    p: NDArray[np.float32],
    triangle: NDArray[np.float32],
    eps: float=1e-6,
) -> NDArray[np.float32]:
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

    uv[uv < 0] = 0

    assert np.all(uv >= 0), 'Found negative UV coordinate'

    return uv


def volumes(
    reference: DentalMesh,
    test: DentalMesh,
    verbose: bool=False,
) -> NDArray[np.float32]:
    signs, intersections = intersect(reference, test)

    vertices, triangles = [], []
    signed_volumes = np.zeros(0)
    vertex_count = 0
    for triangle in reference.triangles:
        # ignore triangles with no correspondence
        if np.any(np.isnan(signs[triangle])):
            signed_volumes = np.concatenate((signed_volumes, [np.nan]))
            continue

        # ignore intersecting triangles
        if not (np.all(signs[triangle] == 1) or np.all(signs[triangle] == -1)):
            signed_volumes = np.concatenate((signed_volumes, [0]))
            continue

        six_vertices = np.concatenate((
            reference.vertices[triangle],
            intersections[triangle],
        ))
        pcd = open3d.geometry.PointCloud(
            points=open3d.utility.Vector3dVector(six_vertices),
        )
        mesh, _ = pcd.compute_convex_hull()
        signed_volume = signs[triangle[0]] * mesh.get_volume()
        signed_volumes = np.concatenate((signed_volumes, [signed_volume]))

        if verbose:
            vertices.append(np.asarray(mesh.vertices))
            triangles.append(np.asarray(mesh.triangles) + vertex_count)
            vertex_count += vertices[-1].shape[0]

    if verbose:
        mesh = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(np.concatenate(vertices)),
            triangles=open3d.utility.Vector3iVector(np.concatenate(triangles)),
        )
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.compute_vertex_normals()
        open3d.visualization.draw_geometries([mesh])
        
    return signed_volumes


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
    edges: NDArray[np.int64],
    intersections: List[NDArray[np.float32]],
    init_idx: int,
    eps: float=1e-5
) -> DentalMesh:
    num_extra_vertices = sum([a.shape[0] - 1 for a in intersections])
    extra_vertices, extra_normals = np.zeros((2, num_extra_vertices, 3))
    extra_triangles = np.zeros((num_extra_vertices + len(edges), 3), dtype=int)
    # extra_vertices, extra_normals = np.zeros((2, 0, 3))
    # extra_triangles = np.zeros((0, 3), dtype=int)

    vertex_idx, triangle_idx = 0, 0
    for edge, inters in zip(edges, intersections):
        start_idx, end_idx = edge
        edge_is_degenerate = False

        for k, inter in enumerate(inters[:-1]):
            if np.linalg.norm(inter - inters[k + 1]) < eps:
                continue

            extra_vertices[vertex_idx] = inter
            extra_normals[vertex_idx] = surface.normals[start_idx]

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
                extra_triangles[triangle_idx] = [end_idx, start_idx, init_idx + vertex_idx]
                triangle_idx += 1

            edge_is_degenerate |= triangle_is_degenerate
            if edge_is_degenerate:
                extra_triangles[triangle_idx] = [init_idx + vertex_idx, init_idx + vertex_idx + 1, end_idx]
            else:
                extra_triangles[triangle_idx] = [start_idx, init_idx + vertex_idx, init_idx + vertex_idx + 1]

            vertex_idx += 1
            triangle_idx += 1

        if not edge_is_degenerate:
            extra_triangles[triangle_idx] = [end_idx, start_idx, init_idx + vertex_idx]
            triangle_idx += 1

    extra_vertices = extra_vertices[:vertex_idx]
    extra_normals = extra_normals[:vertex_idx]
    extra_triangles = extra_triangles[:triangle_idx]
    
    # special case for last triangle
    if end_idx == edges[0, 0]:
        extra_triangles[extra_triangles == init_idx + vertex_idx] = init_idx
    else:
        assert False
        extra_triangles = extra_triangles[-2:]

    mesh = DentalMesh(
        vertices=extra_vertices,
        triangles=extra_triangles,
        normals=extra_normals,
        check=False,
    )

    return mesh


def compute_triangle_edges_map(
    intersections: List[NDArray[np.float32]],
    test_triangle_idxs: List[NDArray[np.int64]],
    init_idx: int,
    eps: float=1e-5,
    out: Optional[Dict[int, NDArray[np.float32]]]=None,
) -> Dict[
    int,
    Tuple[NDArray[np.int64], NDArray[np.float32]],
]:
    i = 0
    out = {} if out is None else out
    for triangle_idxs, inters in zip(test_triangle_idxs, intersections):
        inters = np.stack((inters[:-1], inters[1:])).transpose(1, 0, 2)
        for key, inters in zip(triangle_idxs, inters):
            add = np.linalg.norm(inters[0] - inters[1]) >= eps

            if key not in out:
                out[key] = (np.ones(0, dtype=int), np.ones((0, 3)))

            vertex_idxs = [i + init_idx, i + init_idx + add]
            out[key] = (
                np.concatenate((out[key][0], vertex_idxs)),
                np.concatenate((out[key][1], inters)),
            )            
            
            i += add

    if test_triangle_idxs[0][0] == test_triangle_idxs[-1][-1]:
        key_copy = key
        
        over_idx = out[key][0].max()
        for key in out:
            out[key][0][out[key][0] == over_idx] = init_idx

        prev_key = triangle_idxs[-2]
        prev_triangle_idx = out[prev_key][0][-1]

        prev_triangle_mask = out[key_copy][0] == prev_triangle_idx
        current_start_idx = prev_triangle_mask.shape[0] - 1 - prev_triangle_mask[::-1].argmax()
        current_start_idx = 2 * int(current_start_idx // 2)  # make it even

        out[key_copy] = tuple([
            np.concatenate((
                out[key_copy][i][current_start_idx:],
                out[key_copy][i][:current_start_idx],
            )) for i in range(2)
        ])
    else:
        assert False

    return out


def orient_triangles(
    vertices: NDArray[np.float32],
    triangles: NDArray[np.int64],
    normal: NDArray[np.float32],
) -> NDArray[np.int64]:
    triangle_vertices = vertices[triangles]
    triangle_vectors1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    triangle_vectors2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    triangle_normals = np.cross(triangle_vectors1, triangle_vectors2)

    upside_down = normal @ triangle_normals.T < 0
    triangles[upside_down] = triangles[upside_down][:, [1, 0, 2]]

    return triangles


def delaunay_triangulation(
    triangle_vertices: NDArray[np.float32],
    points: NDArray[np.float32],
    internal_edges: NDArray[np.int64],
    verbose: bool=False,
):  
    diffs = triangle_vertices[:, np.newaxis] - points[np.newaxis]
    distances = np.linalg.norm(diffs, axis=-1)
    points = np.concatenate((triangle_vertices, points))
    uvs = barycentric_coords(points, triangle_vertices)

    is_vertex = np.all(uvs == 0, axis=-1) | np.any(uvs == 1, axis=-1)

    uvs = np.concatenate((uvs[:3], uvs[3:][~is_vertex[3:]]))
    points = np.concatenate((points[:3], points[3:][~is_vertex[3:]]))
    edge_mask = np.all(~is_vertex[3:][internal_edges], axis=-1)
    internal_edges = internal_edges[edge_mask]

    
    uvws = np.column_stack((uvs[3:], 1 - uvs[3:, 0] - uvs[3:, 1]))
    is_border = np.any(uvws < 1e-8, axis=-1)

    border_idxs, border_orders = np.full((2, uvws.shape[0]), fill_value=-1)
    border_idxs[is_border] = uvws[is_border].argmin(axis=-1)

    border0 = (border_idxs == 0) + uvws[:, 1]
    offset = border0.shape[0] - (border_idxs == 0).sum()
    border0 = border0.argsort()[offset:]
    border0 = np.concatenate(([2], 3 + border0, [1]))

    border1 = (border_idxs == 1) + uvws[:, 0]
    offset = border1.shape[0] - (border_idxs == 1).sum()
    border1 = border1.argsort()[offset:]
    border1 = np.concatenate(([2], 3 + border1, [0]))

    border2 = (border_idxs == 2) + uvws[:, 0]
    offset = border2.shape[0] - (border_idxs == 2).sum()
    border2 = border2.argsort()[offset:]
    border2 = np.concatenate(([1], 3 + border2, [0]))

    tri_dict = {
        'vertices': uvs,
        'segments': np.concatenate((
            3 + internal_edges,
            np.column_stack((border0[:-1], border0[1:])),
            np.column_stack((border1[:-1], border1[1:])),
            np.column_stack((border2[:-1], border2[1:])),
        )),
    }

    equal_uvs = np.all(uvs[3:, np.newaxis] == uvs[np.newaxis, 3:], axis=-1)
    if equal_uvs.sum() == uvs.shape[0] - 3:
        # only use triangle package after check, otherwise crash
        t = triangulate(tri_dict, 'p')
        vertices, triangles = t['vertices'], t['triangles']

        if triangles.shape[0] <= 3:
            return triangles, -1

        edges = np.concatenate((
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )) - 3

        internal_edges = internal_edges[internal_edges[:, 0] != internal_edges[:, 1]]
        has_edge = internal_edges[:, np.newaxis] == edges[np.newaxis]
        has_edge = np.any(np.all(has_edge, axis=-1), axis=-1)

        if np.all(has_edge):
            # standard Delaunay triangulation is sufficient
            return triangles, -1

    if verbose:
        lc = mc.LineCollection(uvs[tri_dict['segments']])
        plt.gca().add_collection(lc)
        plt.gca().margins(0.1)
        plt.show(block=True)
    
    tri = Delaunay(uvs)
    triangles = tri.simplices
    triangle_copy = triangles.copy()  

    print('non-manifold!')

    if verbose:
        plt.triplot(uvs[:, 0], uvs[:, 1], triangles)
        plt.title(f'Number of triangles: {triangles.shape[0]}')
        plt.show(block=True)

    uvws = np.column_stack((uvs[3:], 1 - uvs[3:, 0] - uvs[3:, 1]))
    border_idxs, border_orders = np.full((2, uvws.shape[0]), fill_value=-1)
    is_border = np.any(uvws < 1e-8, axis=-1)
    border_idxs[is_border] = uvws[is_border].argmin(axis=-1)
    border_orders[border_idxs == 0] = uvws[border_idxs == 0, 1].argsort().argsort()
    border_orders[border_idxs == 1] = uvws[border_idxs == 1, 0].argsort().argsort()
    border_orders[border_idxs == 2] = uvws[border_idxs == 2, 0].argsort().argsort()
    unique, counts = np.unique(border_idxs, return_counts=True)
    border_counts = np.zeros(3, dtype=int)
    border_counts[unique[unique >= 0]] = counts[unique >= 0]

    triangles = np.zeros((0, 3), dtype=int)
    for i, (start_idx, end_idx) in enumerate(internal_edges):
        if (
            (i > 0 and internal_edges[i - 1, 1] == internal_edges[i, 0])
            or border_idxs[start_idx] == -1
        ):
            idx = end_idx
        else:
            idx = start_idx

        border_idx = border_idxs[idx]

        if border_idx == -1:
            continue

        border_order = border_orders[idx]

        if border_order > 0:
            w = (border_idxs == border_idx) & (border_orders == (border_order - 1))
            prev_vertex_idx = 3 + w.argmax()
        else:
            prev_vertex_idx = min(2, 3 - border_idx)
        triangle = [end_idx + 3, start_idx + 3, prev_vertex_idx]
        triangles = np.concatenate((triangles, [triangle]))
        
        if border_order < border_counts[border_idx] - 1:
            w = (border_idxs == border_idx) & (border_orders == (border_order + 1))
            next_vertex_idx = 3 + w.argmax()
        else:
            next_vertex_idx = max(0, 1 - border_idx)
        triangle = [start_idx + 3, end_idx + 3, next_vertex_idx]
        triangles = np.concatenate((triangles, [triangle]))

    # add triangles between internal vertices
    idxs = np.nonzero(border_idxs == -1)[0] + 3
    # if idxs.shape[0] < 2 or np.linalg.norm(uvs[idxs[0]] - uvs[idxs[1]]) >= 1e-4:
    #     edges = np.sort(np.concatenate((
    #         triangles[:, [0, 1]],
    #         triangles[:, [1, 2]],
    #         triangles[:, [2, 0]],
    #     )), axis=-1)
    #     for idx in idxs:
    #         inside_edges = edges[np.any(edges == idx, axis=-1)]
    #         unique, counts = np.unique(inside_edges, return_counts=True, axis=0)
    #         triangle = np.unique(unique[counts == 1])

    #         if np.any(triangle < 3):
    #             continue

    #         triangles = np.concatenate((triangles, [triangle]))

    uvws = np.column_stack((uvs, 1 - uvs[:, 0] - uvs[:, 1]))
    if idxs.shape[0] < 2 or np.linalg.norm(uvs[idxs[0]] - uvs[idxs[1]]) >= 1e-4:
        edges = np.sort(np.concatenate((
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )), axis=-1)
        for idx in idxs:
            _, inverse, counts = np.unique(edges, return_inverse=True, return_counts=True, axis=0)
            inside_mask = np.any(edges == idx, axis=-1) & (counts == 1)[inverse]

            for idx1, idx2 in edges[inside_mask]:
                if idx2 == idx:
                    idx1, idx2 = idx2, idx1

                inside_mask2 = ~inside_mask & np.any(edges == idx2, axis=-1) & (counts == 1)[inverse]
                inside_edges = edges[inside_mask2]
                for edge in inside_edges:
                    if np.any(np.all(uvws[edge] < 1e-4, axis=0)):
                        continue

                    triangle = np.sort(np.concatenate((edge, [idx])))
                    triangles = np.concatenate((triangles, [triangle]))

    # add triangle with vertex not yet included
    mask = np.unique(triangles)[np.newaxis] == np.arange(points.shape[0])[:, np.newaxis]
    missing = ~np.any(mask, axis=-1)
    missing_idxs = np.nonzero(missing)[0]
    if missing_idxs.shape[0] > 0:
        if missing_idxs[0] == 0:
            if 1 in border_idxs:
                first_idx = 3 + ((border_idxs == 1) + border_orders).argmax()
            else:
                first_idx = 2
            
            if 2 in border_idxs:
                second_idx = 3 + ((border_idxs == 2) + border_orders).argmax()
            else:
                second_idx = 1
        elif missing_idxs[0] == 1:
            if 0 in border_idxs:
                first_idx = 3 + ((border_idxs == 0) + border_orders).argmax()
            else:
                first_idx = 2
            
            if 2 in border_idxs:
                second_idx = 3 + ((border_idxs != 2) + border_orders).argmin()
            else:
                second_idx = 0
        elif missing_idxs[0] == 2:
            if 0 in border_idxs:
                first_idx = 3 + ((border_idxs != 0) + border_orders).argmin()
            else:
                first_idx = 1
            
            if 1 in border_idxs:
                second_idx = 3 + ((border_idxs != 1) + border_orders).argmin()
            else:
                second_idx = 0

        triangle = np.array([missing_idxs[0], first_idx, second_idx])
        triangles = np.concatenate((triangles, [triangle]))

    # add any remaining triangle
    edges = np.sort(np.concatenate((
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]],
    )), axis=-1)

    cross_mask = ~np.any(np.all(uvws[edges] < 1e-5, axis=1), axis=-1)
    unique, counts = np.unique(edges[cross_mask], return_counts=True, axis=0)
    remaining_cross_edges = unique[counts == 1]

    border_edges = np.sort(tri_dict['segments'][internal_edges.shape[0]:], axis=-1)
    mask = np.all(edges[np.newaxis] == border_edges[:, np.newaxis], axis=-1)
    remaining_border_edges = border_edges[~np.any(mask, axis=-1)]

    remaining_edges = np.concatenate((remaining_cross_edges, remaining_border_edges))
    if remaining_border_edges.shape[0] > 0:
        mask = np.any(remaining_cross_edges == remaining_border_edges[0], axis=-1)
        remaining_cross_edges = remaining_cross_edges[mask]
        
        triangle = np.unique(np.concatenate((remaining_border_edges, remaining_cross_edges)))
        triangles = np.concatenate((triangles, [triangle]))    
        
    # only keep unique triangles
    triangles = np.sort(triangles, axis=-1)
    triangles = np.unique(triangles, axis=0)

    diffs = uvs[triangles][np.newaxis, np.newaxis] - uvs[triangles][:, :, np.newaxis, np.newaxis]
    mask = np.all(np.any(np.linalg.norm(diffs, axis=-1) < 1e-5, axis=-1), axis=1)
    same_idxs = np.column_stack(np.nonzero(mask))
    same_idxs = same_idxs[same_idxs[:, 0] != same_idxs[:, 1]]
    assert same_idxs.shape[0] <= 2, 'Unable to make Delaunay triangulation'

    triangle_mask = np.ones(triangles.shape[0], dtype=bool)
    triangle_mask[same_idxs[1:, 0]] = False
    triangles = triangles[triangle_mask]    
    
    # orient triangles consistently
    vertices = points[triangle_copy[0]]
    vector1 = vertices[1] - vertices[0]
    vector2 = vertices[2] - vertices[0]
    normal = np.cross(vector1, vector2)

    triangles = orient_triangles(points, triangles, normal)

    if verbose:
        plt.triplot(uvs[:, 0], uvs[:, 1], triangles)
        plt.title(f'Number of triangles: {triangles.shape[0]}')
        plt.show(block=True)

    inside_idxs = np.nonzero(border_idxs == -1)[0]
        
    return triangles, inside_idxs[0]


def determine_triangles_from_test(
    surface: DentalMesh,
    test: DentalMesh,
    triangle_edges_map: Dict[
        int,
        Tuple[NDArray[np.int64], NDArray[np.float32]],
    ],
    verbose: bool=False,
) -> NDArray[np.int64]:
    extra_triangles = []
    test_triangle_idxs = np.array(list(triangle_edges_map.keys()))
    for test_triangle_idx in test_triangle_idxs:
        vertex_idxs, intersections = triangle_edges_map[test_triangle_idx]

        unique, index, inverse = np.unique(
            vertex_idxs, return_index=True, return_inverse=True,
        )

        if test_triangle_idx in [17175]:
            k = 3

        # determine internal triangles of test triangle
        test_triangle = test.triangles[test_triangle_idx]
        try:
            triangles, duplicate_idx = delaunay_triangulation(
                test.vertices[test_triangle],
                intersections[index],
                inverse.reshape(-1, 2),
                # verbose=verbose,
            )
        except Exception:
            triangles = np.zeros((0, 3), dtype=int)

        # get rid of non-manifold vertices by making a duplicate vertex
        # if duplicate_idx != -1:
        #     extra_vertices = np.concatenate((extra_vertices, []))
        
        # translate vertex indices to accomodate surface and test triangles
        triangles = triangles - 3
        triangles[triangles >= 0] = unique[triangles[triangles >= 0]]
        triangles[triangles < 0] = test_triangle[triangles[triangles < 0] + 3]
        triangles[triangles < test.num_vertices] += surface.num_vertices

        if np.any(triangles == 376):
            k = 3

        extra_triangles.append(triangles)

    triangles = np.concatenate(extra_triangles)
    triangles = triangles[:, [1, 0, 2]]
    triangles = np.unique(triangles, axis=0)

    return triangles


def insert_loops(
    boundaries: List[NDArray[np.int64]],
) -> List[NDArray[np.int64]]:
    keep_idxs = list(range(len(boundaries)))
    for i, edges1 in enumerate(boundaries):
        for j, edges2 in enumerate(boundaries[i + 1:], i + 1):
            mask = edges1[np.newaxis, :, 0] == edges2[:, 0, np.newaxis]
            if not np.any(mask):
                continue

            edges1_idx = np.nonzero(np.any(mask, axis=0))[0][0]
            edges1 = np.concatenate((edges1[edges1_idx:], edges1[:edges1_idx]))

            edges2_idx = np.nonzero(np.any(mask, axis=1))[0][0]
            edges2 = np.concatenate((
                edges2[:edges2_idx], edges1, edges2[edges2_idx:],
            ))

            boundaries[j] = edges2
            keep_idxs.remove(i)

    boundaries = [edges for i, edges in enumerate(boundaries) if i in keep_idxs]

    return boundaries


def compute_boundaries(
    surface: DentalMesh,
) -> List[NDArray[np.int64]]:    
    edges = np.sort(surface.edges, axis=-1)
    _, index, counts = np.unique(
        edges, return_index=True, return_counts=True, axis=0,
    )
    edge_indices = index[counts == 1]

    border_edges = surface.edges[edge_indices]
    triangle_idxs = np.arange(surface.edges.shape[0])[edge_indices]
    triangle_idxs = triangle_idxs % surface.triangles.shape[0]

    boundaries = []
    tails, heads = np.zeros((2, 0), dtype=int)
    for edge, triangle_idx in zip(border_edges, triangle_idxs):
        if np.any(tails == edge[1]):
            idx1 = np.argmax(tails == edge[1])
            boundaries[idx1] = [edge] + boundaries[idx1]

            if boundaries[idx1][0][0] == boundaries[idx1][-1][1]:
                tails[idx1] = -1
                heads[idx1] = -1
                continue

            if not np.any(heads == edge[0]):
                tails[idx1] = edge[0]
                continue
            
            idx2 = np.argmax(heads == edge[0])
            boundaries[idx2] = boundaries[idx2] + boundaries[idx1]
            heads[idx2] = heads[idx1]

            boundaries = boundaries[:idx1] + boundaries[idx1 + 1:]
            tails = np.concatenate((tails[:idx1], tails[idx1 + 1:]))
            heads = np.concatenate((heads[:idx1], heads[idx1 + 1:]))
            continue

        if np.any(heads == edge[0]):
            idx1 = np.argmax(heads == edge[0])
            boundaries[idx1] = boundaries[idx1] + [edge]

            if boundaries[idx1][0][0] == boundaries[idx1][-1][1]:
                tails[idx1] = -1
                heads[idx1] = -1
                continue

            if not np.any(tails == edge[1]):
                heads[idx1] = edge[1]
                continue
            
            idx2 = np.argmax(tails == edge[1])
            boundaries[idx2] = boundaries[idx1] + boundaries[idx2]
            tails[idx2] = tails[idx1]

            boundaries = boundaries[:idx1] + boundaries[idx1 + 1:]
            tails = np.concatenate((tails[:idx1], tails[idx1 + 1:]))
            heads = np.concatenate((heads[:idx1], heads[idx1 + 1:]))
            continue

        boundaries.append([edge])
        tails = np.concatenate((tails, [edge[0]]))
        heads = np.concatenate((heads, [edge[1]]))
        
    boundaries = [np.stack(edges) for edges in boundaries]
    boundaries = insert_loops(boundaries)

    return boundaries   


def plane_lines_intersections(
    plane_eq: NDArray[np.float32],
    lines: NDArray[np.float32],
    eps: float=1e-5,
) -> NDArray[np.float32]:
    u = lines[:, 1] - lines[:, 0]
    dot = plane_eq[:3] @ u.T

    p_co = -plane_eq[:3] * plane_eq[3]
    w = lines[:, 0] - p_co
    fac = -plane_eq[:3] @ w.T / dot
    intersections = lines[:, 0] + (u * fac[:, np.newaxis])

    outside = (
        (intersections < lines.min(axis=1) - eps)
        | (lines.max(axis=1) + eps < intersections)
    )
    outside = np.any(outside, axis=-1)
    intersections[outside] = np.inf

    return intersections


def plane_equation(
    p0: NDArray[np.float32],
    vector1: NDArray[np.float32],
    vector2: NDArray[np.float32],
) -> NDArray[np.float32]:
    plane_normal = np.cross(vector1, vector2)
    plane_normal /= np.linalg.norm(plane_normal)
    plane_d = p0 @ plane_normal
    plane_eq = np.concatenate((plane_normal, [-plane_d]))

    return plane_eq
    
    

def compute_edge_triangle_intersections(
    surface: DentalMesh,
    test: DentalMesh,
    surface_edges: NDArray[np.int64],
) -> List[NDArray[np.int64]]:
    closest_points = surface.closest_points(test)
    test_edges = np.sort(test.edges, axis=-1)

    intersections, test_triangle_idxs = [], []
    next_triangle_idx = -1
    prev_on_border = False
    for edge in surface_edges:
        if np.all(edge == [78, 77]):
            k = 3
        edge_vertices = closest_points['points'][edge]
        edge_triangle_idxs = closest_points['primitive_ids'][edge]

        if 256 in edge_triangle_idxs:
            k = 3

        if edge_triangle_idxs[0] == edge_triangle_idxs[1]:
            # surface edge projects to single test triangle
            intersections.append(list(edge_vertices))
            test_triangle_idxs.append(list(edge_triangle_idxs))

            continue

        # add stats of first test point
        intersections.append([edge_vertices[0]])
        test_triangle_idxs.append([edge_triangle_idxs[0]])

        # determine plane going through edge
        edge_vector = edge_vertices[1] - edge_vertices[0]
        edge_vector /= np.linalg.norm(edge_vector)
        plane_eq = plane_equation(
            p0=edge_vertices[0],
            vector1=edge_vector,
            vector2=surface.vertices[edge[0]] - edge_vertices[0],
        )

        current_triangle_idx = edge_triangle_idxs[0]
        stop = False
        prev_edge = np.full(2, fill_value=-1)
        i = 0
        while True:
            if i > 1000:

                print(current_triangle_idx)
                k = 3

            # determine intersections of edge plane with current triangle
            triangle = test.triangles[current_triangle_idx]
            triangle_edges = np.stack((
                triangle[:2], triangle[1:], triangle[[2, 0]],
            ))
            triangle_edges = np.sort(triangle_edges, axis=-1)


            tri_intersections = plane_lines_intersections(
                plane_eq, test.vertices[triangle_edges],
            )

            # determine test edge index of next intersection along surface edge
            intersection_vectors = tri_intersections - intersections[-1][-1]
            distances = np.linalg.norm(intersection_vectors, axis=-1)

            if (distances < 1e-4).sum() == 2:
                # next is on vertex
                intersect_idxs = np.nonzero(distances < 1e-4)[0]
                on_vertex = True
            else:
                # next is on edge
                norm_mask = (distances >= 1e-4) & np.isfinite(distances)
                intersection_vectors[norm_mask] /= distances[norm_mask, np.newaxis]

                directions = np.zeros_like(distances)
                directions[norm_mask] = edge_vector @ intersection_vectors[norm_mask].T

                intersect_idxs = ((directions <= 0) + distances).argsort()[:1]
                on_vertex = False

            for idx in intersect_idxs:
                next_edge = np.sort(triangle_edges[idx], axis=-1)
                if on_vertex and np.all(prev_edge == next_edge):
                    continue

                # determine neighbouring triangle
                edge_mask = np.all(test_edges == next_edge, axis=-1)
                triangle_idxs = np.nonzero(edge_mask)[0] % test.triangles.shape[0]
                next_intersection = tri_intersections[idx]
                
                if np.any(np.isinf(next_intersection)):
                    k = 3

                if np.linalg.norm(next_intersection - [3.45506001, 12.44924164,-20.48722458]) < 1e-4:
                    k = 3

                break

            if (
                not stop and
                (on_vertex or np.all(directions <= 0))
            ):
                # remove previous point if previous point is on edge/vertex
                intersections[-1] = intersections[-1][:-1]
                test_triangle_idxs[-1] = test_triangle_idxs[-1][:-1]

            if i == 0 and not prev_on_border and (on_vertex or np.all(directions <= 0)):
                intersections[-2] = intersections[-2][:-1]
                test_triangle_idxs[-2] = test_triangle_idxs[-2][:-1]
                next_triangle_idx = current_triangle_idx
                

            if triangle_idxs.shape[0] == 1:
                # encounter a border
                next_triangle_idx = -1
                break

            # determine next triangle
            current_triangle_idx = triangle_idxs[triangle_idxs != current_triangle_idx][0]

            if current_triangle_idx == 17124:
                k = 3
            prev_edge = next_edge

            if (
                i == 0 and
                next_triangle_idx >= 0 and
                current_triangle_idx != next_triangle_idx and
                edge_triangle_idxs[0] != next_triangle_idx and
                (distances < 1e-4).sum() == 1
            ):
                if next_triangle_idx == 17128:
                    k = 3

                # add triangle if only one point is involved
                intersections[-2].append(intersections[-2][-1])
                test_triangle_idxs[-2].append(next_triangle_idx)
                intersections[-1].insert(0, intersections[-2][-1])
                test_triangle_idxs[-1].insert(0, next_triangle_idx)
                next_intersection = intersections[-2][-1]

            # save stats
            if stop:
                next_triangle_idx = current_triangle_idx
                break
            else:
                intersections[-1].append(next_intersection)
                test_triangle_idxs[-1].append(current_triangle_idx)

            if current_triangle_idx == edge_triangle_idxs[1]:
                stop = True

            i += 1

        if (
            not (  # do keep final test triangle idx
                np.all(edge == surface_edges[-1]) and
                test_triangle_idxs[0][0] == test_triangle_idxs[-1][-1]
            ) and
            np.linalg.norm(intersections[-1][-1] - edge_vertices[1]) < 1e-5
        ):
            # remove last point if last point is on edge/vertex
            next_triangle_idx = test_triangle_idxs[-1][-1]
            test_triangle_idxs[-1] = test_triangle_idxs[-1][:-1]
            prev_on_border = True
        else:
            # add stats of last point
            prev_on_border = False
            intersections[-1].append(edge_vertices[1])

    intersections = [np.stack(inters) for inters in intersections]
    test_triangle_idxs = [np.stack(idxs) for idxs in test_triangle_idxs]
            
    return intersections, test_triangle_idxs


def crop_watertight(
    mesh: DentalMesh,
    verbose: bool=False,
) -> DentalMesh:
    # # fill any missing holes
    # o3d_mesh = mesh.to_open3d_triangle_mesh()
    # o3dt_mesh = open3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
    # o3dt_mesh = o3dt_mesh.fill_holes()
    # triangles = o3dt_mesh.triangle.indices.numpy()

    # # extra_triangles = triangles[mesh.triangles.shape[0]:]
    # extra_triangles = triangles[np.any(triangles >= total_vertices, axis=-1)]
    # unique, counts = np.unique(extra_triangles, return_counts=True, axis=0)
    # extra_triangles = unique[counts == 1]

    # mesh.triangles = np.concatenate((mesh.triangles, extra_triangles))


    while True:
        edges = np.sort(mesh.edges, axis=-1)
        _, inverse, counts = np.unique(
            edges, return_inverse=True, return_counts=True, axis=0,
        )

        if counts.min() == 2:
            break

        is_border_edge = (counts == 1)[inverse]

        edge_idxs = np.arange(edges.shape[0])
        border_triangle_idxs = edge_idxs[is_border_edge] % mesh.triangles.shape[0]
        inside_triangle_mask = np.ones(mesh.triangles.shape[0], dtype=bool)
        inside_triangle_mask[border_triangle_idxs] = False
        
        inside_vertex_idxs = np.unique(mesh.triangles[inside_triangle_mask])
        inside_vertex_mask = np.zeros(mesh.vertices.shape[0], dtype=bool)
        inside_vertex_mask[inside_vertex_idxs] = True
        
        if verbose:
            outside_vertex_idxs = np.unique(mesh.triangles[~inside_triangle_mask])
            outside_vertex_mask = np.ones(mesh.vertices.shape[0], dtype=bool)
            outside_vertex_mask[outside_vertex_idxs] = False
            mesh.labels = 1 - outside_vertex_mask.astype(int)
            o3d_mesh = mesh.to_open3d_triangle_mesh()
            open3d.io.write_triangle_mesh('test2.obj', o3d_mesh)
            open3d.visualization.draw_geometries([o3d_mesh])

        mesh.triangles = mesh.triangles[inside_triangle_mask]
        mesh = mesh[inside_vertex_mask]

        if mesh.vertices.shape[0] == 0:
            break

    return mesh    


def repair_self_intersecting_triangles(
    mesh: DentalMesh,
) -> DentalMesh:
    o3d_mesh = mesh.to_open3d_triangle_mesh()
    triangle_pairs = np.asarray(o3d_mesh.get_self_intersecting_triangles())

    if triangle_pairs.shape[0] == 0:
        return mesh
    
    triangle_idxs = np.unique(triangle_pairs)
    vertex_idxs = np.unique(mesh.triangles[triangle_idxs])
    remove_mask = np.any(mesh.triangles[None] == vertex_idxs[:, None, None], axis=(0, 2))

    mesh.triangles = mesh.triangles[~remove_mask]
    o3d_mesh = mesh.to_open3d_triangle_mesh()
    o3d_mesh = open3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
    o3d_mesh = o3d_mesh.fill_holes().to_legacy()

    triangles = np.asarray(o3d_mesh.triangles)
    triangles[mesh.triangles.shape[0]:] = triangles[mesh.triangles.shape[0]:, [1, 0, 2]]
    mesh.triangles = triangles

    return mesh


def split_triangles(mesh: open3d.geometry.TriangleMesh) -> open3d.geometry.TriangleMesh:
    """
    Split the mesh in independent triangles    
    """
    triangles = np.asarray(mesh.triangles).copy()
    vertices = np.asarray(mesh.vertices).copy()

    triangles_3 = np.zeros_like(triangles)
    vertices_3 = np.zeros((len(triangles) * 3, 3), dtype=vertices.dtype)

    for index_triangle, t in enumerate(triangles):
        index_vertex = index_triangle * 3
        vertices_3[index_vertex] = vertices[t[0]]
        vertices_3[index_vertex + 1] = vertices[t[1]]
        vertices_3[index_vertex + 2] = vertices[t[2]]

        triangles_3[index_triangle] = np.arange(index_vertex, index_vertex + 3)

    mesh_return = deepcopy(mesh)
    mesh_return.triangles = open3d.utility.Vector3iVector(triangles_3)
    mesh_return.vertices = open3d.utility.Vector3dVector(vertices_3)
    return mesh_return


def make_holes(
    surface: DentalMesh,
    boundaries: List[NDArray[np.int64]],
    mesh: DentalMesh,
) -> DentalMesh:
    if len(boundaries) == 1:
        return mesh
    
    extra_vertices = mesh.vertices[surface.vertices.shape[0]:]
    extra_triangles_mask = np.all(mesh.triangles >= surface.vertices.shape[0], axis=-1)
    extra_triangles = mesh.triangles[extra_triangles_mask]
    triangle_idx_map = np.arange(mesh.triangles.shape[0])
    triangle_idx_map = triangle_idx_map[extra_triangles_mask]

    boundary_counts = [len(edges) for edges in boundaries]
    boundary_idxs = np.argsort(boundary_counts)[:-1]

    triangle_idxs = []
    for idx in boundary_idxs:
        vertex_idxs = boundaries[idx][:, 0]
        centroid = surface.vertices[vertex_idxs].mean(axis=0)

        vertex_idx = np.linalg.norm(extra_vertices - centroid, axis=-1).argmin()
        vertex_idx += surface.vertices.shape[0]

        triangle_idx = np.any(extra_triangles == vertex_idx, axis=-1).argmax()
        triangle_idx = triangle_idx_map[triangle_idx]

        triangle_idxs.append(triangle_idx)

    triangle_mask = np.ones(mesh.triangles.shape[0], dtype=bool)
    triangle_mask[triangle_idxs] = False

    mesh.triangles = mesh.triangles[triangle_mask]

    return mesh


def duplicate_non_manifold_vertices(
    mesh: DentalMesh,
    verbose: bool=False,
) -> DentalMesh:
    o3d_mesh = mesh.to_open3d_triangle_mesh()
    non_manifold_vertex_idxs = np.asarray(o3d_mesh.get_non_manifold_vertices())

    if verbose:
        labels = np.zeros(mesh.num_vertices, dtype=int)
        labels[non_manifold_vertex_idxs] = 1
        mesh.labels = labels

        open3d.visualization.draw_geometries([mesh.to_open3d_triangle_mesh()])


    for idx in non_manifold_vertex_idxs:
        triangle_mask = np.any(mesh.triangles == idx, axis=-1)

        triangles = mesh.triangles[triangle_mask]
        vertex_idxs = np.unique(triangles)

        if verbose:
            vertex_mask = np.zeros(mesh.num_vertices, dtype=bool)
            vertex_mask[vertex_idxs] = True
            vertex_map = np.cumsum(vertex_mask) - 1

            mesh2 = mesh[vertex_mask]
            mesh2.triangles = vertex_map[triangles]
            
            open3d.visualization.draw_geometries([mesh2.to_open3d_triangle_mesh()])

        matrix = mesh.vertices[vertex_idxs, np.newaxis] - mesh.vertices[np.newaxis, vertex_idxs]
        distances = np.linalg.norm(matrix, axis=-1)
        same_idxs = np.column_stack(np.nonzero(distances < 1e-5))

        same_idxs = same_idxs[same_idxs[:, 0] != same_idxs[:, 1]]
        if same_idxs.shape[0] == 0:
            continue


        same_idxs = vertex_idxs[same_idxs[0]]

        group1, group2 = np.array([same_idxs[0]]), np.array([same_idxs[1]])
        work_triangle_idxs = np.arange(triangles.shape[0]).tolist()
        triangle_idxs1, triangle_idxs2 = [], []
        while work_triangle_idxs:
            j = 0
            for triangle_idx in work_triangle_idxs.copy():
                triangle = triangles[triangle_idx]

                if np.any(triangle[np.newaxis] == group1[:, np.newaxis]):
                    group1 = np.concatenate((group1, triangle[triangle != idx]))
                    work_triangle_idxs = work_triangle_idxs[:j] + work_triangle_idxs[j + 1:]
                    triangle_idxs1.append(triangle_idx)
                    continue

                if np.any(triangle[np.newaxis] == group2[:, np.newaxis]):
                    group2 = np.concatenate((group2, triangle[triangle != idx]))
                    work_triangle_idxs = work_triangle_idxs[:j] + work_triangle_idxs[j + 1:]
                    triangle_idxs2.append(triangle_idx)
                    continue

                j += 1

        if verbose:
            vertex_mask = np.zeros(mesh.num_vertices, dtype=bool)
            vertex_mask[vertex_idxs] = True
            mesh2 = mesh[vertex_mask]

            vertex_map = np.cumsum(vertex_mask) - 1
            mesh2.triangles = vertex_map[triangles]

            vertex_inverse_map = np.zeros(vertex_idxs.max() + 1, dtype=int)
            vertex_inverse_map[vertex_idxs] = np.arange(vertex_idxs.shape[0])
            vertex_idxs1 = vertex_inverse_map[group1]
            vertex_idxs2 = vertex_inverse_map[group2]
            
            vertex_mask = np.zeros(mesh2.vertices.shape[0], dtype=bool)
            vertex_mask[vertex_idxs2] = True
            mesh2.labels = vertex_mask.astype(int)
            
            open3d.visualization.draw_geometries([mesh2.to_open3d_triangle_mesh()])
        
        triangle_idxs2 = np.array(triangle_idxs2)
        current_triangle_mask = np.zeros(triangles.shape[0], dtype=bool)
        current_triangle_mask[triangle_idxs2] = True

        triangle_mask[triangle_mask] = current_triangle_mask
        non_manifold_vertex_mask = triangle_mask[:, np.newaxis] & (mesh.triangles == idx)

        centroid = mesh.vertices[np.unique(mesh.triangles[triangle_mask])].mean(axis=0)

        new_surface_vertex = mesh.vertices[idx] - 1e-4 * (mesh.vertices[idx] - centroid)
        new_test_vertex = mesh.vertices[same_idxs[1]] - 1e-4 * (mesh.vertices[same_idxs[1]] - centroid)

        mesh.vertices[same_idxs[1]] = new_test_vertex
        mesh.triangles[non_manifold_vertex_mask] = mesh.vertices.shape[0]

        mesh.vertices = np.concatenate((mesh.vertices, [new_surface_vertex]))
        mesh.normals = np.concatenate((mesh.normals, [mesh.normals[idx]]))
        if mesh.labels is not None:
            mesh.labels = np.concatenate((mesh.labels, [mesh.labels[idx]]))

    return mesh


def determine_extra_triangles(
    surface: DentalMesh,
    test: DentalMesh,
    edges: NDArray[np.int64],
    init_idx: int,
) -> Tuple[DentalMesh, NDArray[np.int64]]:
    # determine intersections of surface edges with test triangles
    intersections, test_triangle_idxs = compute_edge_triangle_intersections(
        surface, test, edges,
    )

    # determine extra triangles from surface triangles
    vertices, surface_triangles, normals = determine_triangles_from_surface(
        surface, edges, intersections, init_idx,
    )

    # determine extra triangles from test triangles
    test_triangles, test_triangle_idxs = determine_triangles_from_test(
        test, intersections, test_triangle_idxs, init_idx,
    )
    test_triangles[test_triangles < init_idx] += surface.vertices.shape[0]

    triangles = np.concatenate((surface_triangles, test_triangles))
    mesh = DentalMesh(vertices, triangles, normals, check=False)

    test_triangle_idxs = np.concatenate((
        np.full(surface_triangles.shape[0], fill_value=-1), test_triangle_idxs,
    ))

    return mesh, test_triangle_idxs


def fill_holes(
    mesh: DentalMesh
) -> DentalMesh:
    dists = np.linalg.norm(mesh.vertices, axis=-1)

    o3d_mesh = mesh.to_open3d_triangle_mesh()
    o3d_mesh = open3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
    o3d_mesh = o3d_mesh.fill_holes(hole_size=1.0)
    o3d_mesh = o3d_mesh.to_legacy()

    vertices = np.asarray(o3d_mesh.vertices)
    distances = np.linalg.norm(vertices, axis=-1)
    vertex_mask = distances < dists.max() * 1.2

    triangles = np.asarray(o3d_mesh.triangles)
    triangle_mask = np.all(vertex_mask[triangles], axis=-1)

    vertex_map = np.cumsum(vertex_mask) - 1
    triangles = vertex_map[triangles]

    return DentalMesh(
        vertices=vertices[vertex_mask],
        triangles=triangles,
        normals=np.asarray(o3d_mesh.vertex_normals)[vertex_mask],
    )


def surface_volume(
    surface: DentalMesh,
    test: DentalMesh,
    verbose: bool=True,
    positive: bool=False,
) -> DentalMesh:
    if verbose:
        open3d.visualization.draw_geometries([surface.to_open3d_triangle_mesh()])

    vertices, triangles, normals = [], [], []
    init_idx = surface.num_vertices + test.num_vertices
    triangle_mask = np.ones(test.num_triangles, dtype=bool)
    boundaries = compute_boundaries(surface)
    triangle_edges_map = {}
    for edges in boundaries:
        # determine intersections of surface edges with test triangles
        intersections, test_triangle_idxs = compute_edge_triangle_intersections(
            surface, test, edges,
        )
        triangle_mask[np.concatenate(test_triangle_idxs)] = False

        # determine extra triangles from surface triangles
        mesh = determine_triangles_from_surface(
            surface, edges, intersections, init_idx,
        )
        vertices.append(mesh.vertices)
        triangles.append(mesh.triangles)
        normals.append(mesh.normals)

        triangle_edges_map = compute_triangle_edges_map(
            intersections, test_triangle_idxs, init_idx,
            out=triangle_edges_map,
        )

        init_idx += mesh.num_vertices


    test_triangles = determine_triangles_from_test(
        surface, test, triangle_edges_map,
    )    
    mesh = DentalMesh(
        vertices=np.concatenate((surface.vertices, test.vertices, *vertices)),
        triangles=np.concatenate((
            surface.triangles,
            surface.num_vertices + test.triangles[triangle_mask][:, [1, 0, 2]],
            *triangles,
            test_triangles,
        )),
        normals=np.concatenate((surface.normals, test.normals, *normals)),
    )

    if positive:
        mesh.triangles = mesh.triangles[:, [1, 0, 2]]

    # mesh = fill_holes(mesh)
    mesh = make_holes(surface, boundaries, mesh)
    mesh = duplicate_non_manifold_vertices(mesh)
    mesh = crop_watertight(mesh, verbose=False)

    if verbose:
        open3d.visualization.draw_geometries([mesh.to_open3d_triangle_mesh()])

    return mesh


def volumes2(
    reference: DentalMesh,
    test: DentalMesh,
    verbose: bool=False,
) -> NDArray[np.float32]:
    signed_dists = reference.signed_distances(test)
    signed_dists[reference.border_mask(layers=2)] = np.nan
    
    pos = reference[signed_dists > 0.02].crop_components()
    neg = reference[signed_dists < -0.02].crop_components()

    volumes = []
    for surface in tqdm(pos[1:]):
        mesh = surface_volume(surface, test, verbose, positive=True)
        o3d_mesh = mesh.to_open3d_triangle_mesh()
        volumes.append(o3d_mesh.get_volume())

    meshes = []
    for surface in tqdm(neg):
        mesh = surface_volume(surface, test, verbose, positive=False)
        o3d_mesh = mesh.to_open3d_triangle_mesh()

        meshes.append(o3d_mesh)
        volumes.append(o3d_mesh.get_volume())

    open3d.visualization.draw_geometries(meshes)

    return volumes
