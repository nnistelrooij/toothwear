from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
import open3d
from tqdm import tqdm

from toothwear.teeth import DentalMesh
from toothwear.visualization import draw_meshes
from toothwear.volume.boundaries import compute_boundaries
from toothwear.volume.intersections import determine_boundary_test_intersections, TestGapException
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

    is_positive = np.array([True, False]).repeat((len(pos), len(neg)))

    return pos + neg, is_positive


def make_holes(
    surface: DentalMesh,
    boundaries: List[NDArray[np.int64]],
    test_triangle_edges_map: Dict[
        int,
        Tuple[NDArray[np.int64], NDArray[np.float32]],
    ],
    mesh: DentalMesh,
) -> DentalMesh:
    # remove triangles for which internal triangles were added
    test_triangle_idxs = np.array(list(test_triangle_edges_map.keys()))
    test_triangle_idxs = test_triangle_idxs[test_triangle_idxs >= 0]
    triangle_mask = np.ones(mesh.num_triangles, dtype=bool)
    triangle_mask[surface.num_triangles + test_triangle_idxs] = False
    
    mesh.triangles = mesh.triangles[triangle_mask]

    if len(boundaries) == 1:
        return mesh
    
    extra_vertices = mesh.vertices[surface.num_vertices:]
    extra_triangles_mask = np.all(mesh.triangles >= surface.num_vertices, axis=-1)
    extra_triangles = mesh.triangles[extra_triangles_mask]
    triangle_idx_map = np.arange(mesh.num_triangles)
    triangle_idx_map = triangle_idx_map[extra_triangles_mask]

    boundary_counts = [edges.shape[0] for edges in boundaries]
    boundary_idxs = np.argsort(boundary_counts)[:-1]

    triangle_idxs = []
    for idx in boundary_idxs:
        vertex_idxs = boundaries[idx][:, 0]
        centroid = surface.vertices[vertex_idxs].mean(axis=0)

        vertex_idx = np.linalg.norm(extra_vertices - centroid, axis=-1).argmin()
        vertex_idx += surface.num_vertices

        triangle_idx = np.any(extra_triangles == vertex_idx, axis=-1).argmax()
        triangle_idx = triangle_idx_map[triangle_idx]

        triangle_idxs.append(triangle_idx)

    triangle_mask = np.ones(mesh.num_triangles, dtype=bool)
    triangle_mask[triangle_idxs] = False

    mesh.triangles = mesh.triangles[triangle_mask]

    return mesh


def crop_watertight(
    mesh: DentalMesh,
    verbose: bool=False,
) -> DentalMesh:
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
            if np.any(outside_vertex_idxs < 1224):
                k = 3
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


def compute_volume(
    mesh: DentalMesh,
) -> float:
    triangles_vertices = mesh.vertices[mesh.triangles]

    cross = np.cross(triangles_vertices[:, 1], triangles_vertices[:, 2])
    volume = np.einsum('ij,ij', triangles_vertices[:, 0], cross) / 6

    return volume


def determine_mesh_between(
    surface: DentalMesh,
    test: DentalMesh,
    is_positive: bool,
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

        init_idx = np.sort(vertex_idxs[-1])[-1] + 1

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

    if is_positive:
        mesh.triangles = mesh.triangles[:, [1, 0, 2]]
    
    mesh = make_holes(surface, boundaries, test_triangle_edges_map, mesh)
    mesh = crop_watertight(mesh, verbose=False)


    # o3d_mesh = mesh.to_open3d_triangle_mesh()
    # triangle_pairs = np.asarray(o3d_mesh.get_self_intersecting_triangles())

    # if triangle_pairs.shape[0] == 0:
    #     return mesh
    
    # for triangle_pair in triangle_pairs:    
    #     vertex_idxs = np.unique(mesh.triangles[triangle_pair])
    #     vertex_mask = np.zeros(mesh.num_vertices, dtype=bool)
    #     vertex_mask[vertex_idxs] = True
    #     mesh.labels = vertex_mask.astype(int)
    #     open3d.visualization.draw_geometries([mesh.to_open3d_triangle_mesh()])

    return mesh
    

def volumes(
    reference: DentalMesh,
    test: DentalMesh,
    verbose: bool=False
) -> NDArray[np.float64]:
    surfaces, is_positive = crop_positive_negative_surfaces(reference, test)

    volumes, meshes = [], []
    for surface, is_positive in tqdm(
        iterable=zip(surfaces, is_positive),
        desc='Determining meshes',
    ):
        try:
            mesh = determine_mesh_between(surface, test, is_positive, verbose)
            volume = compute_volume(mesh)

            meshes.append(mesh)
            volumes.append(volume if is_positive else -volume)
        except TestGapException:
            print('Cannot close surface')

    o3d_meshes = [mesh.to_open3d_triangle_mesh() for mesh in meshes]
    open3d.visualization.draw_geometries(o3d_meshes)

    return volumes
