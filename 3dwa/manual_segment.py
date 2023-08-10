from pathlib import Path
from typing import List

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

        sgn_dists = arch_mesh.signed_distances(tooth_mesh, ignore_border=False)
        arch_mesh.labels[sgn_dists <= 0.2] = fdi

    return arch_mesh    


def patient_segmentations(
    root: Path, patient: str,
):
    for path in sorted(root.glob(f'{patient}/*')):
        if path.is_file():
            continue

        intake = path.name.split('-')[0]
        intake_files = sorted(path.glob(f'*{intake}.stl'))

        maxilla_file = [f for f in intake_files if 'maxilla' in f.name]
        mandible_file = [f for f in intake_files if 'mandible' in f.name]
        tooth_files = [f for f in intake_files if f.name[0] != 'm']

        if not maxilla_file or not mandible_file:
            continue

        maxilla_mesh = arch_segmentation(maxilla_file[0], tooth_files)
        mandible_mesh = arch_segmentation(mandible_file[0], tooth_files)

        draw_meshes(maxilla_mesh)
        draw_meshes(mandible_mesh)


if __name__ == '__main__':
    root = Path('methodology study')
    for patient in ['A27']:
        patient_segmentations(root, patient)
