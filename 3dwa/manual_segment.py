import json
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

import os, sys
sys.path.append(os.getcwd())

from toothwear.teeth import DentalMesh
from toothwear.visualization import draw_meshes
from toothwear.register import ransac_icp


def arch_segmentation(
    arch_file: Path, tooth_files: List[Path],
):
    arch_mesh = DentalMesh.from_files(arch_file, reference=True)
    for file in tqdm(tooth_files):
        fdi = int(file.name.split('_')[0])        
        if (
            'maxilla' in arch_file.name and fdi > 28 or
            'mandible' in arch_file.name and fdi <= 28
        ):
            continue

        tooth_mesh = DentalMesh.from_files(file, reference=True)
        result = ransac_icp(tooth_mesh, arch_mesh)
        tooth_mesh = tooth_mesh.transform(result.transformation)

        sgn_dists = arch_mesh.signed_distances(tooth_mesh, ignore_border=False, remove_border_clusters=False)
        mask = np.abs(sgn_dists) <= 0.2
        neg_idxs = np.nonzero(~mask)[0]

        neg_mesh = arch_mesh[~mask].to_open3d_triangle_mesh()
        labels, counts, _ = neg_mesh.cluster_connected_triangles()
        for label in range(len(counts)):
            vertex_idxs = np.unique(np.asarray(neg_mesh.triangles)[np.asarray(labels) == label])
            dist = np.abs(sgn_dists[neg_idxs[vertex_idxs]]).mean()

            if dist < 1:
                mask[neg_idxs[vertex_idxs]] = True
                
        
        arch_mesh.labels[mask] = fdi

 
    with open(f'methodology study/3DWA labels/{arch_file.stem}.json', 'w') as f:
        json.dump({'labels': arch_mesh.labels.tolist()}, f, indent=2)

    return arch_mesh    


def patient_segmentations(
    root: Path, patient: str,
):
    for i, path in enumerate(sorted(root.glob(f'3DWA pairs/{patient}/*'))):
        if path.is_file():
            continue

        intake, recall = path.name.split('-')
        tooth_files = sorted(path.glob(f'*{intake}.stl'))

        if i == 0:
            ori_files = sorted((root / 'meshes').glob(f'{patient}_0y*'))
            ori_files += sorted((root / 'meshes').glob(f'{patient.replace("-", "")}_0y*'))
            maxilla_file = [f for f in ori_files if 'maxilla' in f.name]
            mandible_file = [f for f in ori_files if 'mandible' in f.name]
            
            maxilla_mesh = arch_segmentation(maxilla_file[0], tooth_files)
            mandible_mesh = arch_segmentation(mandible_file[0], tooth_files)

        # interval = int(recall) - int(intake)
        # ori_files = sorted((root / 'meshes').glob(f'{patient}_{interval}y*'))
        # ori_files += sorted((root / 'meshes').glob(f'{patient.replace("-", "")}_{interval}y*'))
        # maxilla_file = [f for f in ori_files if 'maxilla' in f.name]
        # mandible_file = [f for f in ori_files if 'mandible' in f.name]

        # if not maxilla_file or not mandible_file:
        #     continue

        # maxilla_mesh = arch_segmentation(maxilla_file[0], tooth_files)
        # mandible_mesh = arch_segmentation(mandible_file[0], tooth_files)

        # draw_meshes(maxilla_mesh)
        # draw_meshes(mandible_mesh)


if __name__ == '__main__':
    root = Path('methodology study')
    for patient in ['A-41', 'A-29', 'A-28', 'A-25', 'A-24', 'A-20']:
        patient_segmentations(root, patient)
