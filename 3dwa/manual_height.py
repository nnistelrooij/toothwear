from pathlib import Path
from typing import List

import numpy as np
import open3d
import pandas as pd

import os, sys
sys.path.append(os.getcwd())

from toothwear.teeth import DentalMesh
from toothwear.visualization import draw_heatmap


fdis = [
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48,
]


def pair_heights(
    intake_files: List[Path],
    followup_files: List[Path],
    invert_x: bool=False,
):
    colored_teeth, max_heights = {}, {}
    for ref_file, test_file in zip(intake_files, followup_files):
        if ref_file.name[0] == 'm':
            continue

        ref_mesh = DentalMesh.from_files(ref_file, reference=True)
        test_mesh = DentalMesh.from_files(test_file, reference=False)



        draw_heatmap(ref_mesh, test_mesh, verbose=False)
        draw_heatmap(test_mesh, ref_mesh, verbose=False)

        
        colored_tooth, max_dist = draw_heatmap(ref_mesh, test_mesh, return_max=True)

        if invert_x:
            colored_tooth.vertices = open3d.utility.Vector3dVector(np.column_stack((
                -np.asarray(colored_tooth.vertices)[:, :1],
                np.asarray(colored_tooth.vertices)[:, 1:],
            )))
            colored_tooth.vertex_normals = open3d.utility.Vector3dVector(np.column_stack((
                -np.asarray(colored_tooth.vertex_normals)[:, :1],
                np.asarray(colored_tooth.vertex_normals)[:, 1:],
            )))

        fdi = int(ref_file.name.split('_')[0])
        colored_teeth[fdi] = colored_tooth
        max_heights[fdi] = max_dist

    return colored_teeth, max_heights


def patient_heights(root: Path, patient: str, verbose: bool=False):
    df = pd.DataFrame({'FDI': fdis})
    for path in sorted(root.glob(f'{patient}/*'))[::-1]:
        if path.is_file():
            continue

        intake, followup = path.name.split('-')
        if 'operator' in followup:
            followup = followup.split('_')[0]
        intake_files = sorted(path.glob(f'*{intake}.stl'))
        followup_files = sorted(path.glob(f'*{followup}.stl'))

        colored_teeth, max_heights = pair_heights(intake_files, followup_files)

        if verbose:
            # save colored teeth for external inspection
            for fdi, o3d_mesh in colored_teeth.items():
                open3d.io.write_triangle_mesh(str(intake_files[0].parent / f'{fdi}.stl'), o3d_mesh)

            # visualize colored teeth of maxilla and mandible
            open3d.visualization.draw_geometries(
                [mesh for fdi, mesh in colored_teeth.items() if fdi <= 28]
            )
            open3d.visualization.draw_geometries(
                [mesh for fdi, mesh in colored_teeth.items() if fdi > 28]
            )

        # save most negative distance of each tooth
        col_name = f'0-{int(followup) - int(intake)}'
        if 'operator' in path.name:
            col_name += '_' + path.name.split('_')[1]
        df[col_name] = [max_heights.get(fdi, {'max': ''})['max'] for fdi in fdis]

    return df


if __name__ == '__main__':
    root = Path('methodology study/3DWA pairs')
    verbose = False

    dfs = {}
    # for patient in ['A-46', 'A-24', 'A-29', 'A-20', 'A-25', 'A-28', 'A-41', 'A-27']:
    for patient in ['A-02', 'A-20', 'A-40', 'A-47']:
        print('Patient', patient)
        df = patient_heights(root, patient, verbose=verbose)
        dfs[patient] = df

    with pd.ExcelWriter(root / '3dwa_heights_double.xlsx') as writer:
        for patient, df in dfs.items():
            df.to_excel(writer, sheet_name=patient, index=False)
