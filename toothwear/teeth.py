import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import open3d
import pandas as pd
import pymeshlab


palette = np.array([
    [174, 199, 232],
    [152, 223, 138],
    [31, 119, 180],
    [255, 187, 120],
    [188, 189, 34],
    [140, 86, 75],
    [255, 152, 150],
    [214, 39, 40],
    [197, 176, 213],
    [148, 103, 189],
    [196, 156, 148], 
    [23, 190, 207], 
    [247, 182, 210], 
    [219, 219, 141], 
    [255, 127, 14], 
    [158, 218, 229], 
    [44, 160, 44], 
    [112, 128, 144], 
    [227, 119, 194], 
    [82, 84, 163],
    [100, 100, 100],
], dtype=np.uint8)


class DentalMesh:

    def __init__(
        self,
        vertices: Optional[NDArray[np.float64]]=None,
        triangles: Optional[NDArray[np.int64]]=None,
        normals: Optional[NDArray[np.float64]]=None,
        labels: Optional[NDArray[np.int64]]=None,
        reference: bool=True,
        check: bool=True,
    ) -> None:
        if vertices is None:
            vertices = np.ones((0, 3))
            
        if normals is None:
            normals = np.ones((0, 3))

        if triangles is None:
            triangles = np.ones((0, 3), dtype=int)

        triangles = triangles.astype(np.int64)

        if check:
            # remove unreferenced vertices
            vertex_idxs = np.unique(triangles)
            vertex_mask = np.zeros(vertices.shape[0], dtype=bool)
            vertex_mask[vertex_idxs] = True
        else:
            vertex_mask = np.ones(vertices.shape[0], dtype=bool)
        vertices = vertices[vertex_mask].astype(np.float64)
        normals = normals[vertex_mask].astype(np.float64)

        # realign vertex indices
        if check:
            vertex_map = np.cumsum(vertex_mask) - 1
            triangles = vertex_map[triangles]

        # normalize normals
        norms = np.linalg.norm(normals, axis=-1)
        normals[norms > 0] /= norms[norms > 0, np.newaxis]

        # set default labels
        if labels is None:
            labels = np.zeros(vertices.shape[0], dtype=int)
        else:
            labels = labels[vertex_mask].astype(np.int64)

        self.vertices = vertices
        self.triangles = triangles
        self.normals = normals
        self.labels = labels
        self.reference = reference

    @classmethod
    def from_files(
        cls,
        mesh_file: Union[Path, str],
        ann_file: Optional[Union[Path, str]]=None,
        **kwargs,
    ):
        vertices, triangles, normals = DentalMesh.load_mesh(mesh_file)
        labels = DentalMesh.load_labels(ann_file) if ann_file else None

        return cls(vertices, triangles, normals, labels, **kwargs)

    def close_holes(
        self,
        maxholesize: int=130,
    ):
        mesh = pymeshlab.Mesh(
            vertex_matrix=self.vertices,
            face_matrix=self.triangles,
            v_normals_matrix=self.normals,
        )

        ms = pymeshlab.MeshSet()
        ms.add_mesh(mesh)
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_close_holes(maxholesize=maxholesize)

        mesh = ms.current_mesh()
        mesh.compact()
        vertices = mesh.vertex_matrix()
        triangles = mesh.face_matrix()
        normals = mesh.vertex_normal_matrix()

        return DentalMesh(
            vertices, triangles, normals, reference=self.reference,
        )        
        
    @staticmethod
    def load_mesh(
        mesh_file: Union[Path, str],
    ) -> open3d.geometry.TriangleMesh:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(mesh_file))
        # ms.meshing_repair_non_manifold_edges()
        # ms.meshing_close_holes(maxholesize=130)
        
        mesh = ms.current_mesh()
        vertices = mesh.vertex_matrix()
        triangles = mesh.face_matrix()
        normals = mesh.vertex_normal_matrix()

        return vertices, triangles, normals
    
    @staticmethod
    def load_labels(
        ann_file: Union[Path, str],
    ) -> NDArray[np.int64]:
        with open(ann_file, 'r') as f:
            ann_dict = json.load(f)

        labels = ann_dict['labels']
        labels = np.array(labels)

        # correct FDI confusion between arches
        if 'mandible' in str(ann_file) and np.any(labels[labels > 0] <= 28):
            labels[labels > 0] += 20
        elif 'maxilla' in str(ann_file) and np.any(labels[labels > 0] > 28):
            labels[labels > 0] -= 20        

        return labels
    
    def copy(self):
        return DentalMesh(
            self.vertices.copy(),
            self.triangles.copy(),
            self.normals.copy(),
            self.labels.copy(),
            self.reference,
        )
    
    def to_open3d_triangle_mesh(
        self,
        colors: bool=True,
    ) -> open3d.geometry.TriangleMesh:
        o3d_mesh = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(self.vertices),
            triangles=open3d.utility.Vector3iVector(self.triangles),
        )
        o3d_mesh.compute_vertex_normals()

        if colors and self.labels is not None:
            _, classes = np.unique(self.labels, return_inverse=True)
            colors = palette[classes - 1]
        else:
            colors = palette[-1:].repeat(self.vertices.shape[0], axis=0)
        o3d_mesh.vertex_colors = open3d.utility.Vector3dVector(colors / 255)

        return o3d_mesh
    
    def to_open3d_point_cloud(self) -> open3d.geometry.PointCloud:
        pcd = open3d.geometry.PointCloud(
            points=open3d.utility.Vector3dVector(self.vertices),
        )
        pcd.normals = open3d.utility.Vector3dVector(self.normals)

        return pcd
    
    def transform(
        self,
        T: NDArray[np.float32],
    ):
        vertices = np.column_stack(
            (self.vertices, np.ones(self.vertices.shape[0])),
        )
        vertices = vertices @ T.T
        vertices = vertices[:, :3]

        normals = self.normals @ T[:3, :3].T

        return DentalMesh(
            vertices=vertices,
            triangles=self.triangles,
            normals=normals,
            labels=self.labels,
            reference=self.reference,
        )        
    
    def _subsample_triangles(
        self,
        vertex_mask: NDArray[np.bool_],
    ) -> NDArray[np.int64]:
        triangle_mask = np.all(vertex_mask[self.triangles], axis=-1)
        triangles = self.triangles[triangle_mask]

        vertex_idxs = np.unique(triangles)
        vertex_mask = np.zeros_like(vertex_mask)
        vertex_mask[vertex_idxs] = True

        vertex_map = np.cumsum(vertex_mask) - 1
        triangles = vertex_map[triangles]

        return vertex_mask, triangles
    
    def __getitem__(self, mask: NDArray[np.bool_]):
        mask, triangles = self._subsample_triangles(mask)
        vertices = self.vertices[mask]
        normals = self.normals[mask]
        labels = self.labels[mask] if self.labels is not None else None

        return DentalMesh(vertices, triangles, normals, labels, self.reference)
    
    def crop_teeth(self) -> Dict[int, Any]:
        teeth = {}
        for label in self.unique_labels:
            mask = self.labels == label
            tooth = self[mask]
            # tooth = tooth.close_holes()
            teeth[label] = tooth

        return teeth
    
    def crop_wrt_normal(
        self,
        normal: NDArray[np.float32],
        ratio: float=0.8,
        return_mask: bool=False,
    ):
        directions = self.normals @ normal
        index = int(directions.shape[0] * (1 - ratio))
        direction_thresh = np.sort(directions)[index]
        mask = directions >= direction_thresh
        
        out = self[mask].crop_largest_component()

        if return_mask:
            mask, _ = self._subsample_triangles(mask)
            mask[mask] = self[mask].crop_largest_component(return_mask=True)
            
            return out, mask        
        
        return out
    
    def _ray_triangle_intersections(
        self,
        scene: open3d.t.geometry.RaycastingScene,
        allow_back_faces: bool=False,
        max_dist: Optional[float]=2.0,
    ) -> NDArray[np.float32]:
        rays = np.stack((
            np.column_stack((self.vertices, self.normals)),
            np.column_stack((self.vertices, -self.normals)),
        ))
        ray_dict = scene.cast_rays(rays.astype(np.float32))
        ray_dict = {k: v.numpy() for k, v in ray_dict.items()}

        if not allow_back_faces:
            hit = np.isfinite(ray_dict['t_hit'])
            signs = np.tile([[1], [-1]], reps=hit.shape[1])
            normals = signs[..., None] * ray_dict['primitive_normals']
            backward = np.einsum('ik,ik->i', normals[hit], rays[hit, 3:]) < 0
            hit[hit] = ~backward
            ray_dict['t_hit'][~hit] = np.inf

        if max_dist is not None:
            hit = np.isfinite(ray_dict['t_hit'])
            hit[hit] = np.abs(ray_dict['t_hit'][hit]) < max_dist
            ray_dict['t_hit'][~hit] = np.inf

        signs = 1 - 2 * ray_dict['t_hit'].argmin(axis=0)
        distances = ray_dict['t_hit'].min(axis=0).astype(np.float64)
        
        hit = np.isfinite(distances)[:, np.newaxis]
        idx = ray_dict['t_hit'].argmin(axis=0), np.arange(hit.shape[0]), slice(None)
        closest_points = {
            'signed_distances': np.where(hit[:, 0], signs * distances, np.nan),
            'primitive_normals': np.where(hit, ray_dict['primitive_normals'][idx], np.nan),
            'primitive_uvs': np.where(hit, ray_dict['primitive_uvs'][idx], np.nan),
            'primitive_ids': np.where(hit[:, 0], ray_dict['primitive_ids'][idx[:-1]], -1),
            'geometry_ids': np.where(hit[:, 0], ray_dict['geometry_ids'][idx[:-1]], -1),
        }

        return closest_points
    
    def closest_points(
        self,
        test,
        raycast: bool=False,
        remove_border_clusters: bool=True,
    ) -> Dict[str, NDArray[Any]]:
        test = test.to_open3d_triangle_mesh()
        test = open3d.t.geometry.TriangleMesh.from_legacy(test)

        scene = open3d.t.geometry.RaycastingScene()
        scene.add_triangles(test)

        if raycast:
            ray_dict = self._ray_triangle_intersections(scene)
            signed_dists = ray_dict['signed_distances'][:, np.newaxis]
            ray_dict['points'] = self.vertices + signed_dists * self.normals

            return ray_dict

        closest_points = scene.compute_closest_points(self.vertices.astype(np.float32))
        closest_points = {k: v.numpy() for k, v in closest_points.items()}

        test_points = closest_points['points'].astype(np.float64)
        vectors = test_points - self.vertices
        distances = np.linalg.norm(vectors, axis=-1)

        # determine sign of distance
        directions = np.einsum('ik,ik->i', self.normals, vectors)
        signs = np.sign(directions)
        closest_points['signed_distances'] = signs * distances

        if not remove_border_clusters:
            return closest_points

        # make measurements for non-corresponding vertices np.nan
        ray_dict = self._ray_triangle_intersections(scene)
        nohit = np.isnan(ray_dict['signed_distances'])
        nohit, _ = self._subsample_triangles(nohit)

        if not np.any(nohit):
            return closest_points

        crop = self[nohit]
        clusters = crop.to_open3d_triangle_mesh().cluster_connected_triangles()
        cluster_idxs = np.asarray(clusters[0])

        is_border_vertex = self.border_mask(layers=1)
        is_border_triangle = np.any(is_border_vertex[nohit][crop.triangles], -1)
        is_border_cluster = np.zeros(cluster_idxs.max() + 1, dtype=bool)
        for i in range(cluster_idxs.max() + 1):
            is_border_cluster[i] = np.any(is_border_triangle[cluster_idxs == i])

        border_triangles = crop.triangles[is_border_cluster[cluster_idxs]]
        border_vertices = np.unique(border_triangles)
        vertex_mask = np.zeros(crop.vertices.shape[0], dtype=bool)
        vertex_mask[border_vertices] = True
        nohit[nohit] = vertex_mask

        # remove points from triangles with too fast changes
        triangle_mask = closest_points['signed_distances'][self.triangles].ptp(1) < 1.0
        vertex_mask = np.ones(self.vertices.shape[0], dtype=bool)
        vertex_mask[self.triangles[~triangle_mask]] = False
        nohit = nohit | ~vertex_mask

        closest_points['points'] = closest_points['points'].astype(np.float64)
        closest_points['points'][nohit] = np.nan
        closest_points['geometry_ids'][nohit] = -1
        closest_points['primitive_ids'][nohit] = -1
        closest_points['primitive_uvs'][nohit] = np.nan
        closest_points['primitive_normals'][nohit] = np.nan
        closest_points['signed_distances'][nohit] = np.nan

        return closest_points        
    
    def signed_distances(
        self,
        test,
        ignore_border: bool=True,
        remove_border_clusters: bool=True,
    ) -> NDArray[np.float32]:
        closest_points = self.closest_points(test, remove_border_clusters=remove_border_clusters)

        sgn_dists = closest_points['signed_distances']
        if ignore_border:
            sgn_dists[self.border_mask()] = np.nan

        return sgn_dists
    
    def signed_volumes(
        self,
        test,
        volume_thresh: float=0.2**3,
        verbose: bool=False,
    ) -> NDArray[np.float32]:
        # determining corresponding poinst between self and test
        closest_points = self.closest_points(test, raycast=False)
        signs = np.sign(closest_points['signed_distances'])
        intersections = closest_points['points']

        vertices, triangles = [], []
        volumes = np.zeros(0)
        vertex_count = 0
        for triangle in self.triangles:
            # ignore triangles with no correspondence
            if np.any(np.isnan(signs[triangle])):
                volumes = np.concatenate((volumes, [np.nan]))
                continue

            # ignore intersecting triangles
            if not (np.all(signs[triangle] == 1) or np.all(signs[triangle] == -1)):
                volumes = np.concatenate((volumes, [0]))
                continue

            six_vertices = np.concatenate((
                self.vertices[triangle],
                intersections[triangle],
            ))
            pcd = open3d.geometry.PointCloud(
                points=open3d.utility.Vector3dVector(six_vertices),
            )
            mesh, _ = pcd.compute_convex_hull()
            volumes = np.concatenate((volumes, [mesh.get_volume()]))

            if verbose:
                vertices.append(np.asarray(mesh.vertices))
                triangles.append(np.asarray(mesh.triangles) + vertex_count)
                vertex_count += vertices[-1].shape[0]
            
        signs = signs[self.triangles][:, 0]
        below_thresh_mask = np.isnan(volumes) | (volumes < volume_thresh)
        signed_volumes = np.where(below_thresh_mask, signs * volumes, 0)

        if verbose:
            print(f'Triangle coverage: {(signed_volumes > 0).mean():.3f}')

            mesh = open3d.geometry.TriangleMesh(
                vertices=open3d.utility.Vector3dVector(np.concatenate(vertices)),
                triangles=open3d.utility.Vector3iVector(np.concatenate(triangles)),
            )
            mesh = mesh.remove_duplicated_vertices()
            mesh = mesh.compute_vertex_normals()
            open3d.visualization.draw_geometries([mesh])
        
        return signed_volumes
    
    def crop_wrt_wear(
        self,
        other,
        ratio: float=0.8,
        return_mask: bool=False,
        crop_largest_component: bool=False,
    ):
        distances = self.signed_distances(other)

        if self.reference:  # reference - test
            index = int(np.sum(~np.isnan(distances)) * (1 - ratio))
            distance_thresh = np.sort(distances)[index]
            mask = distances >= distance_thresh
        else:  # test - reference
            index = int(np.sum(~np.isnan(distances)) * ratio)
            distance_thresh = np.sort(distances)[index]
            mask = distances < distance_thresh

        if return_mask:
            mask, _ = self._subsample_triangles(mask)
            return mask
        
        out = self[mask]
        
        if crop_largest_component:
            return out.crop_largest_component()
        
        return out
    
    def border_mask(
        self,
        layers: int=2,
    ) -> NDArray[np.bool_]:
        mask = np.ones(self.vertices.shape[0], dtype=bool)
        for _ in range(layers):
            edges = self[mask].edges
            edges = np.sort(edges, axis=-1)
            unique, counts = np.unique(edges, return_counts=True, axis=0)
            border_vertex_idxs = np.unique(unique[counts == 1])

            border_vertex_mask = np.zeros(mask.sum(), dtype=bool)
            border_vertex_mask[border_vertex_idxs] = True

            mask[mask] = ~border_vertex_mask
            if layers > 1:
                mask, _ = self._subsample_triangles(mask)

        return ~mask
    
    def border_edges(
        self,
    ) -> NDArray[np.int64]:
        edges = np.sort(self.edges, axis=-1)
        _, index, counts = np.unique(
            edges, return_index=True, return_counts=True, axis=0,
        )
        
        return self.edges[index[counts == 1]]
    
    def crop_components(
        self,
        mask: Optional[NDArray[np.bool_]]=None,
        return_mask: bool=False,
    ) -> List[Any]:
        mesh = self.to_open3d_triangle_mesh()
        cluster_idxs, _, _ = mesh.cluster_connected_triangles()
        cluster_idxs = np.asarray(cluster_idxs)

        components, component_masks = [], []
        for i in range(cluster_idxs.max() + 1):
            triangles = self.triangles[cluster_idxs == i]
            vertex_idxs = np.unique(triangles)
            component_mask = np.zeros(self.vertices.shape[0], dtype=bool)
            component_mask[vertex_idxs] = True
            component_mask, _ = self._subsample_triangles(component_mask)
            if mask is not None:
                component_mask[~mask] = False
            
            components.append(self[component_mask])
            component_masks.append(component_mask)

        if return_mask:
            return component_masks
            
        return components
    
    def crop_largest_component(
        self,
        return_mask: bool=False,
    ):
        mesh = self.to_open3d_triangle_mesh()
        cluster_idxs, counts, _ = mesh.cluster_connected_triangles()
        triangle_mask = cluster_idxs == np.array(counts).argmax()

        triangles = self.triangles[triangle_mask]        
        vertex_idxs = np.unique(triangles)

        mask = np.zeros(self.vertices.shape[0], dtype=bool)
        mask[vertex_idxs] = True

        if return_mask:
            return mask
        
        return self[mask]       
    
    def measure_wear(
        self,
        other,
        normal: NDArray[np.float32],
        wear_thresh: float=0.02,
        direction_thresh: float=0.1,
        area_thresh: float=0.4,
        volume: bool=False,
    ) -> Tuple[int, float, float]:
        # only keep vertices with wear
        distances = self.signed_distances(other)
        mask = ~np.isnan(distances) & (distances < -wear_thresh)
        mask, _ = self._subsample_triangles(mask)

        # remove 2 layers from the boundary
        mask[mask] = ~self[mask].border_mask()
        mask, _ = self._subsample_triangles(mask)

        # determine connected components
        component_masks = self[mask].crop_components(return_mask=True)

        # determine statistics of each component
        columns = ('min', 'mean', 'std', 'direction', 'area')
        columns += ('volume',) if volume else ()
        df = pd.DataFrame(columns=columns)
        for comp_mask in component_masks:
            comp_distances = distances[mask][comp_mask]

            component = self[mask][comp_mask]
            comp_directions = component.normals @ normal

            df.loc[len(df)] = [
                comp_distances.min(),
                comp_distances.mean(),
                comp_distances.std(),
                comp_directions.mean(),
                component.area,
                *((component.get_volume_to(other),) if volume else ()),
            ]

        # determine relevant components
        df['keep'] = (
            (df['std'] >= wear_thresh / 2) &
            (df['direction'] >= direction_thresh) &
            (df['area'] >= area_thresh)
        )

        if not df['keep'].any():
            return -1, 0.0, 0.0
        
        # determine mask for final wear component
        df['heuristic'] = df['keep'] - df['std'] * (4*df['min'] + df['mean'])
        mask[mask] = component_masks[df['heuristic'].argmax()]

        # take minimum measurement of final component
        wear_distances = distances[mask]
        wear_idx, wear_mm = wear_distances.argmin(), wear_distances.min()

        # translate vertex index back to self mesh
        idx_map = np.arange(self.vertices.shape[0])
        idx_map = idx_map[mask]
        wear_idx = idx_map[wear_idx]

        if not volume:
            return wear_idx, wear_mm, 0.0

        return wear_idx, wear_mm, df['volume'][df['keep']].sum()
    
    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_triangles(self) -> int:
        return self.triangles.shape[0]

    @property
    def edges(self) -> NDArray[np.int64]:
        edges = np.concatenate((
            self.triangles[:, :2],
            self.triangles[:, 1:],
            self.triangles[:, [2, 0]],
        ))

        return edges
    
    @property
    def centroid(self) -> NDArray[np.float32]:
        return self.vertices[self.labels > 0].mean(axis=0)
    
    @property
    def unique_labels(self) -> Set[int]:
        labels = np.unique(self.labels)[1:]
        labels = set(labels.tolist())

        return labels
    
    @property
    def centroids(self) -> Dict[int, NDArray[np.float32]]:
        centroids = {}
        for label in self.unique_labels:
            mask = self.labels == label
            tooth_vertices = self.vertices[mask]
            centroid = tooth_vertices.mean(axis=0)

            centroids[label] = centroid

        return centroids
    
    @property
    def area(self) -> float:
        return self.to_open3d_triangle_mesh().get_surface_area()
    
    def get_volume_to(
        self,
        other,
    ) -> float:
        import toothwear.volume.volume2 as volume

        try:
            mesh = volume.determine_mesh_between(self, other, False)
            return volume.compute_volume(mesh)
        except volume.TestGapException:
            return 0.0

    def save(
        self,
        filename: Union[Path, str],
    ) -> None:
        np.savez(
            filename,
            vertices=self.vertices,
            triangles=self.triangles,
            normals=self.normals,
            labels=self.labels,
            reference=self.reference,
        )

    @staticmethod
    def load(
        filename: Union[Path, str]
    ):
        arr_dict = np.load(filename)

        return DentalMesh(
            vertices=arr_dict['vertices'],
            triangles=arr_dict['triangles'],
            normals=arr_dict['normals'],
            labels=arr_dict['labels'],
            reference=arr_dict['reference'],
        )
    