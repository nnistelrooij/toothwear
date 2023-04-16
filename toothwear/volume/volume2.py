from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
import open3d
from tqdm import tqdm

from toothwear.teeth import DentalMesh
from toothwear.visualization import draw_meshes
from toothwear.volume.boundaries import compute_boundaries
from toothwear.volume.intersections import determine_boundary_test_intersections
from toothwear.volume.surface_triangles import determine_triangles_from_surface
from toothwear.volume.test_triangles import determine_triangles_from_test
from toothwear.volume.triangle_edges_map import update_test_triangle_edges_map


def crop_positive_negative_surfaces(
    reference: DentalMesh,
    test: DentalMesh,
    thresh: float=0.02,
) -> Tuple[List[DentalMesh], NDArray[np.bool_]]:
    signed_dists = reference.signed_distances(test)
    signed_dists[reference.border_mask(layers=2)] = np.nan
    
    pos = reference[signed_dists >= thresh].crop_components()
    neg = reference[signed_dists <= -thresh].crop_components()

    return pos + neg


def make_holes(
    surface: DentalMesh,
    boundaries: List[NDArray[np.int64]],
    test_triangle_edges_map: Dict[
        int,
        Tuple[NDArray[np.int64], NDArray[np.float32]],
    ],
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

    # remove triangles for which internal triangles were added
    test_triangle_idxs = np.array(list(test_triangle_edges_map.keys()))
    triangle_mask[test_triangle_idxs + surface.num_triangles] = False

    mesh.triangles = mesh.triangles[triangle_mask]

    return mesh


def crop_watertight(
    mesh: DentalMesh,
    verbose: bool=True,
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


def determine_mesh_between(
    surface: DentalMesh,
    test: DentalMesh,
    verbose: bool=False,
) -> DentalMesh:
    if verbose:
        draw_meshes(surface)

    boundaries = compute_boundaries(surface)
    init_idx = surface.num_vertices + test.num_vertices

    extra = DentalMesh()
    test_triangle_edges_map = {}
    for edges in boundaries:
        intersections, vertex_idxs, test_triangle_idxs = determine_boundary_test_intersections(
            surface, test, edges, init_idx
        )
        extra = determine_triangles_from_surface(
            surface, extra, edges, intersections, vertex_idxs, init_idx,
        )

        update_test_triangle_edges_map(
            test_triangle_edges_map,
            intersections,
            vertex_idxs,
            test_triangle_idxs,
        )

        init_idx = vertex_idxs[-1][:-1].max() + 1

    extra = determine_triangles_from_test(
        surface, test, extra, test_triangle_edges_map,
    )

    mesh = DentalMesh(
        vertices=np.concatenate((surface.vertices, test.vertices, extra.vertices)),
        triangles=np.concatenate((
            surface.triangles,
            surface.num_vertices + test.triangles[:, [1, 0, 2]],
            extra.triangles,
        )),
        normals=np.concatenate((surface.normals, test.normals, extra.normals)),
    )

    if surface.vertices.mean() < extra.vertices.mean():
        mesh.triangles = mesh.triangles[:, [1, 0, 2]]
    
    mesh = make_holes(surface, boundaries, test_triangle_edges_map, mesh)
    mesh = crop_watertight(mesh)

    k = 3

    
    
    

def volumes(
    reference: DentalMesh,
    test: DentalMesh,
    verbose: bool=False
) -> NDArray[np.float64]:
    surfaces = crop_positive_negative_surfaces(reference, test)

    volumes = []
    for surface in tqdm(surfaces, desc='Determining meshes'):
        mesh = determine_mesh_between(surface, test, verbose)
        o3d_mesh = mesh.to_open3d_triangle_mesh()
        volumes.append(o3d_mesh.get_volume())

    return volumes
