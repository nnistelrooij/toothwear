from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import open3d
from scipy.spatial.transform import Rotation

from toothwear.teeth import DentalMesh, palette


def draw_meshes(
    *dental_meshes: List[Union[DentalMesh, open3d.geometry.TriangleMesh]],
    color: bool=True,
) -> None:
    o3d_meshes = []
    for mesh in dental_meshes:
        if isinstance(mesh, open3d.geometry.TriangleMesh):
            o3d_mesh = mesh
        else:
            o3d_mesh = mesh.to_open3d_triangle_mesh(colors=color)
        o3d_meshes.append(o3d_mesh)

    open3d.visualization.draw_geometries(o3d_meshes)


def draw_correspondences(
    reference: DentalMesh,
    test: DentalMesh,
) -> None:
    centroids = np.concatenate((
        np.stack(list(reference.centroids.values())) - reference.centroid,
        np.stack(list(test.centroids.values())) - test.centroid,
    ))
    o3d_pcd = open3d.geometry.PointCloud(
        points=open3d.utility.Vector3dVector(centroids),
    )
    colors = palette[np.arange(len(reference.unique_labels))] / 255
    o3d_pcd.colors = open3d.utility.Vector3dVector(
        np.concatenate((colors, colors)),
    )

    o3d_ls = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(centroids),
        lines=open3d.utility.Vector2iVector(np.column_stack((
            np.arange(centroids.shape[0] // 2),
            np.arange(centroids.shape[0] // 2, centroids.shape[0]),
        )))
    )

    o3d_meshes = []
    for vector in (
        [0, 1, 0], [1, 0, 0], [0, 0, 1],
    ):
        o3dt_ls = open3d.t.geometry.LineSet.from_legacy(o3d_ls)
        o3dt_mesh = o3dt_ls.extrude_linear(vector)
        o3d_meshes.append(o3dt_mesh.to_legacy())

    open3d.visualization.draw_geometries([o3d_pcd])


def distances_to_colors(
    distances: NDArray[np.float32],
    nominal: Tuple[float, float]=(-0.02, 0.02),
    critical: Tuple[float, float]=(-0.2, 0.2),
    n_segments: int=21,
) -> NDArray[np.float32]:
    assert n_segments % 2 == 1, f'n_segments must be odd, got {n_segments}.'

    n_segments = n_segments // 2 - 1

    # make nominal distances zero
    distances = distances.copy()
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
    mask: Optional[NDArray[np.bool_]]=None,
    verbose: bool=False,
    return_max: bool=False,
) -> Union[
    open3d.geometry.TriangleMesh,
    Tuple[open3d.geometry.TriangleMesh, dict]
]:
    distances = reference.signed_distances(test, ignore_border=False)
    colors = distances_to_colors(distances)

    if mask is not None:
        reference = reference[mask]
        colors = colors[mask]

    out = reference.to_open3d_triangle_mesh()
    out.vertex_colors = open3d.utility.Vector3dVector(colors)

    if verbose:
        open3d.visualization.draw_geometries([out])
    
    if return_max:
        distances = reference.signed_distances(test)
        distances = distances[~np.isnan(distances)]
        wear = {'max': np.min(distances), 'mad': np.mean(np.abs(distances - np.mean(distances)))}
        return out, wear

    return out



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
    point_idx: int=-1,
    normal: Optional[NDArray[np.float32]]=None,
) -> None:
    # make heatmap of reference-test distances
    heatmap = draw_heatmap(reference, test)

    # measure point at most negative distance
    if point_idx == -1:
        sgn_dists = reference.signed_distances(test)
        point_idx = np.nanargmin(sgn_dists)

    # color measurement location black
    colors = np.asarray(heatmap.vertex_colors) * 0.5
    colors[point_idx] = [1.0, 0.2, 0.2]
    heatmap.vertex_colors = open3d.utility.Vector3dVector(colors)

    if normal is None:
        normal = reference.normals.mean(0)

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
