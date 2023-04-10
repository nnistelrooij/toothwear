from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import open3d
from scipy.spatial.transform import Rotation

from toothwear.teeth import DentalMesh


def draw_meshes(
    dental_meshes: List[DentalMesh],
    color: bool=True,
) -> None:
    o3d_meshes = []
    for mesh in dental_meshes:
        o3d_mesh = mesh.to_open3d_triangle_mesh(colors=color)
        o3d_meshes.append(o3d_mesh)

    open3d.visualization.draw_geometries(o3d_meshes)


def distances_to_colors(
    distances: NDArray[np.float32],
    nominal: Tuple[float, float]=(-0.02, 0.02),
    critical: Tuple[float, float]=(-0.2, 0.2),
    n_segments: int=21,
) -> NDArray[np.float32]:
    assert n_segments % 2 == 1, f'n_segments must be odd, got {n_segments}.'

    n_segments = n_segments // 2 - 1

    # make nominal distances zero
    nom_mask = (nominal[0] <= distances) & (distances <= nominal[1])
    distances[~nom_mask] -= np.where(
        np.sign(distances[~nom_mask]) < 0, nominal[0], nominal[1],
    )
    distances[nom_mask] = 0

    # make over-critical distances infinity, otherwise between -1 and 1
    critical = (critical[0] - nominal[0], critical[1] - nominal[1])
    mask = (critical[0] <= distances) & (distances <= critical[1])
    distances[~mask] = np.sign(distances[~mask]) * np.inf
    distances[mask] = (distances[mask] - critical[0]) / (critical[1] - critical[0])
    distances[mask] = 2 * distances[mask] - 1

    # determine segments
    distances[mask] = distances[mask] * n_segments
    distances[mask] = np.trunc(distances[mask])
    distances[mask] = distances[mask].clip(1 - n_segments, n_segments - 1)
    distances[mask] = distances[mask] / n_segments

    # determine colors
    colors = np.empty((distances.shape[0], 3))

    colors[np.isnan(distances)] = [0.2, 0.2, 0.2]

    colors[distances >= 0, 0] = 1
    colors[distances >= 0, 1] = 1 - distances[distances >= 0]
    colors[distances >= 0, 2] = 0

    colors[distances <= 0, 0] = 0
    colors[distances <= 0, 1] = 1 + distances[distances <= 0]
    colors[distances <= 0, 2] = 1

    colors[nom_mask] = [0.0, 1.0, 0.0]

    colors[np.isposinf(distances)] = [0.8, 0.0, 0.0]
    colors[np.isneginf(distances)] = [0.0, 0.0, 0.5]

    return colors


def draw_heatmap(
    reference: DentalMesh,
    test: DentalMesh,
    return_geometry: bool=False,
) -> Optional[open3d.geometry.TriangleMesh]:
    distances = reference.signed_distances(test)
    colors = distances_to_colors(distances)

    reference = reference.to_open3d_triangle_mesh()
    reference.vertex_colors = open3d.utility.Vector3dVector(colors)

    if return_geometry:
        return reference

    open3d.visualization.draw_geometries([reference])


def draw_landmark(
    mesh: DentalMesh,
    point_idx: int,
    return_geometry: bool=False,
) -> Optional[open3d.geometry.TriangleMesh]:
    edges = mesh.edges
    point_idxs = np.unique(edges[np.any(edges == point_idx, axis=1)])

    colors = np.full_like(mesh.vertices, 0.4)
    colors[point_idx] = [0.8, 0.0, 0.0]

    mesh = mesh.to_open3d_triangle_mesh()
    mesh.vertex_colors = open3d.utility.Vector3dVector(colors)

    if return_geometry:
        return mesh

    open3d.visualization.draw_geometries([mesh])


def draw_result(
    reference: DentalMesh,
    test: DentalMesh,
    point_idx: int,
    normal: NDArray[np.float32],
) -> None:
    # make heatmap of reference-test distances
    heatmap = draw_heatmap(reference, test, return_geometry=True)

    # color measurement location black
    colors = np.asarray(heatmap.vertex_colors) * 0.5
    colors[point_idx] = [1.0, 0.2, 0.2]
    heatmap.vertex_colors = open3d.utility.Vector3dVector(colors)

    # draw geometry
    viz = open3d.visualization.Visualizer()
    viz.create_window()
    viz.add_geometry(heatmap)
    viz.update_renderer()
    viz.get_view_control().set_front(normal)
    viz.get_view_control().set_up(
        normal @ Rotation.from_euler('x', 90, degrees=True).as_matrix(),
    )
    viz.run()
