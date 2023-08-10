import numpy as np
from numpy.typing import NDArray
import open3d
reg = open3d.pipelines.registration
from procrustes.rotational import rotational

from toothwear.teeth import DentalMesh


def procrustes(
    reference: DentalMesh,
    test: DentalMesh,
) -> NDArray[np.float32]:
    labels = reference.unique_labels & test.unique_labels
    assert len(labels) > 0, 'No corresponding teeth found.'

    a = [c for l, c in reference.centroids.items() if l in labels]
    b = [c for l, c in test.centroids.items() if l in labels]
    result = rotational(a=np.stack(a), b=np.stack(b), translate=True)

    T1 = np.eye(4)
    T1[:3, 3] = -np.mean(a, axis=0)

    R = np.eye(4)
    R[:3, :3] = result.t.T

    T2 = np.eye(4)
    T2[:3, 3] = np.mean(b, axis=0)
    
    return T2 @ R @ T1


def ransac_icp(
    reference: DentalMesh,
    test: DentalMesh,
    verbose: bool=False,
    voxel_size: float=0.2,
) -> NDArray[np.float32]:
    # transform meshes to downsampled point clouds
    reference_pcd = reference.to_open3d_point_cloud()
    reference_pcd = reference_pcd.voxel_down_sample(voxel_size)
    test_pcd = test.to_open3d_point_cloud()
    test_pcd = test_pcd.voxel_down_sample(voxel_size)

    if verbose:
        open3d.visualization.draw_geometries([reference_pcd, test_pcd])

    # compute curvature point features
    fpfh_cfg = {
        'search_param': open3d.geometry.KDTreeSearchParamHybrid(
            radius=5 * voxel_size, max_nn=100,
        ),
    }
    reference_fpfh = reg.compute_fpfh_feature(reference_pcd, **fpfh_cfg)
    test_fpfh = reg.compute_fpfh_feature(test_pcd, **fpfh_cfg)

    # register point clouds globally given feature correspondences
    ransac_cfg = {
        'mutual_filter': True,
        'max_correspondence_distance': 1.5 * voxel_size,
        'estimation_method': reg.TransformationEstimationPointToPoint(),
        'ransac_n': 4,
        'checkers': [
            reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            reg.CorrespondenceCheckerBasedOnDistance(1.5 * voxel_size),
        ],
        'criteria': reg.RANSACConvergenceCriteria(4000000, 0.9999),
    }
    result = reg.registration_ransac_based_on_feature_matching(
        reference_pcd, test_pcd, reference_fpfh, test_fpfh, **ransac_cfg,
    )

    # register point clouds locally given coordinate correspondences
    icp_cfg = {
        'max_correspondence_distance': voxel_size,
        'init': result.transformation,
        'estimation_method': reg.TransformationEstimationPointToPlane(),
        'criteria': reg.ICPConvergenceCriteria(max_iteration=3000),
    }
    result = reg.registration_icp(reference_pcd, test_pcd, **icp_cfg)

    return result
