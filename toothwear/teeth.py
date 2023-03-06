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
        vertices: NDArray[np.float32],
        triangles: NDArray[np.int64],
        normals: NDArray[np.float32],
        labels: Optional[NDArray[np.int64]]=None,
        reference: bool=True,
    ) -> None:
        self.vertices = vertices
        self.triangles = triangles
        self.normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
        self.labels = labels
        self.reference = reference

    @classmethod
    def from_files(
        cls,
        mesh_file: Union[Path, str],
        ann_file: Union[Path, str],
        **kwargs,
    ):
        vertices, triangles, normals = DentalMesh.load_mesh(mesh_file)
        labels = DentalMesh.load_labels(ann_file)

        return cls(vertices, triangles, normals, labels, **kwargs)
        
    @staticmethod
    def load_mesh(
        mesh_file: Union[Path, str],
    ) -> open3d.geometry.TriangleMesh:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(mesh_file))
        
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

        return labels
    
    def to_open3d_triangle_mesh(
        self,
        colors: bool=True,
    ) -> open3d.geometry.TriangleMesh:
        o3d_mesh = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(self.vertices),
            triangles=open3d.utility.Vector3iVector(self.triangles),
        )
        o3d_mesh.compute_vertex_normals()

        if not colors or self.labels is None:
            return o3d_mesh
        
        _, classes = np.unique(self.labels, return_inverse=True)
        colors = palette[classes - 1] / 255
        o3d_mesh.vertex_colors = open3d.utility.Vector3dVector(colors)

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

        if return_mask:
            mask, _ = self._subsample_triangles(mask)
            return mask

        return self[mask]
    
    def signed_distances(
        self,
        test,
        direction_thresh: float=0.5,
    ) -> NDArray[np.float32]:
        test = test.to_open3d_triangle_mesh()
        test = open3d.t.geometry.TriangleMesh.from_legacy(test)

        scene = open3d.t.geometry.RaycastingScene()
        scene.add_triangles(test)

        query_points = self.vertices.astype(np.float32)
        closest_points = scene.compute_closest_points(query_points)

        test_points = closest_points['points'].numpy()
        vectors = test_points - self.vertices
        distances = np.sqrt(np.sum(vectors ** 2, axis=-1))

        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
        test_normals = closest_points['primitive_normals'].numpy()
        directions = np.einsum('ik,ik->i', vectors, test_normals)
        signs = np.sign(directions)

        # make orthogonal measurements np.nan
        distances[np.abs(directions) < direction_thresh] = np.nan

        return signs * distances
    
    def crop_wrt_wear(
        self,
        other,
        ratio: float=0.8,
        return_mask: bool=False,
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

        return self[mask]
    
    def crop_wrt_border(
        self,
        layers: int=2,
        return_mask: bool=False,
    ):
        mask = np.ones(self.vertices.shape[0], dtype=bool)
        for _ in range(layers):
            edges = self[mask].edges
            _, inverse, counts = np.unique(
                edges, return_inverse=True, return_counts=True, axis=0,
            )
            border_edges_mask = counts[inverse] == 1

            border_vertex_idxs = np.unique(edges[border_edges_mask])
            border_vertex_mask = np.zeros(mask.sum(), dtype=bool)
            border_vertex_mask[border_vertex_idxs] = True

            mask[mask] = mask[mask] & ~border_vertex_mask
            mask, _ = self._subsample_triangles(mask)

        if return_mask:
            return mask

        return self[mask]
    
    def crop_components(
        self,
        return_mask: bool=False,
    ) -> List[Any]:
        mesh = self.to_open3d_triangle_mesh()
        cluster_idxs, counts, _ = mesh.cluster_connected_triangles()
        cluster_idxs = np.asarray(cluster_idxs)

        components, masks = [], []
        for i in range(len(counts)):
            triangles = self.triangles[cluster_idxs == i]
            vertex_idxs = np.unique(triangles)
            mask = np.zeros(self.vertices.shape[0], dtype=bool)
            mask[vertex_idxs] = True
            mask, _ = self._subsample_triangles(mask)
            
            masks.append(mask)
            components.append(self[mask])

        if return_mask:
            return masks
            
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
    ) -> Tuple[int, float]:
        # only keep vertices with wear
        distances = self.signed_distances(other)
        mask = ~np.isnan(distances) & (distances < -wear_thresh)
        mask, _ = self._subsample_triangles(mask)

        # remove 2 layers from the boundary
        mask[mask] = self[mask].crop_wrt_border(return_mask=True)
        mask, _ = self._subsample_triangles(mask)

        # determine connected components
        component_masks = self[mask].crop_components(return_mask=True)

        # determine statistics of each component
        df = pd.DataFrame(columns=('min', 'mean', 'std', 'direction', 'area'))
        for comp_mask in component_masks:
            comp_distances = distances[mask][comp_mask]
            comp_directions = self[mask][comp_mask].normals @ normal

            df.loc[len(df)] = [
                comp_distances.min(),
                comp_distances.mean(),
                comp_distances.std(),
                comp_directions.mean(),
                self[mask][comp_mask].area,
            ]

        # determine relevant components
        df['keep'] = (
            (df['std'] >= wear_thresh / 2) &
            (df['direction'] >= direction_thresh) &
            (df['area'] >= area_thresh)
        )

        if not df['keep'].any():
            return -1, 0.0
        
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

        return wear_idx, wear_mm

    @property
    def edges(self) -> NDArray[np.int64]:
        edges = np.concatenate((
            self.triangles[:, :2],
            self.triangles[:, 1:],
            self.triangles[:, [0, 2]],
        ))

        return np.sort(edges, axis=-1)
    
    @property
    def centroid(self) -> NDArray[np.float32]:
        return self.vertices.mean(axis=0)
    
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
