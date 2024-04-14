from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from toothwear.io import load_dental_mesh, save_dental_mesh
from toothwear.register import (
    procrustes,
    ransac_icp,
)
from toothwear.visualization import (
    draw_correspondences,
    draw_heatmap,
    draw_meshes,
    draw_result,
)
from toothwear.teeth import DentalMesh
# from toothwear.volume.volume2 import volumes


fdis = [
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48,
]


def main(
    root: Path,
    reference_stem: str,
    test_stem: str,
    verbose: bool=False,
) -> Dict[int, Tuple[DentalMesh, DentalMesh]]:
    reference = load_dental_mesh(root, reference_stem, reference=True)
    test = load_dental_mesh(root, test_stem, reference=False)
    if verbose:
        draw_meshes(reference, color=False)
        draw_meshes(test, color=False)
        draw_meshes(reference)
        draw_meshes(test)
        draw_correspondences(reference, test)

    T = procrustes(reference, test)
    reference = reference.transform(T)
    if verbose:
        draw_meshes(reference)

    normal = reference[reference.labels > 0].normals.mean(axis=0)
    reference_teeth = reference.crop_teeth()
    test_teeth = test.crop_teeth()

    fdis = tqdm(list(set(reference_teeth) & set(test_teeth)), leave=False)
    tooth_pairs, wears = {}, {}
    for fdi in fdis:
        fdis.set_description(f'#{fdi}')

        reference_tooth = reference_teeth[fdi]
        test_tooth = test_teeth[fdi]
        if verbose:
            draw_heatmap(reference_tooth, test_tooth, verbose=True)
            draw_heatmap(test_tooth, reference_tooth, verbose=True)

        ref_tooth_crop, ref_mask = reference_tooth.crop_wrt_normal(
            -normal, return_mask=True,
        )
        test_tooth_crop, test_mask = test_tooth.crop_wrt_normal(
            -normal, return_mask=True,
        )
        if verbose:
            draw_heatmap(reference_tooth, test_tooth, ref_mask, verbose=True)
            draw_heatmap(test_tooth, reference_tooth, test_mask, verbose=True)
            draw_meshes(ref_tooth_crop, test_tooth_crop)

        result = ransac_icp(ref_tooth_crop, test_tooth_crop)
        reference_tooth = reference_tooth.transform(result.transformation)
        normal = normal @ T[:3, :3]
        if verbose:
            draw_heatmap(reference_tooth, test_tooth, verbose=True)

        ref_tooth_crop = reference_tooth.crop_wrt_wear(test_tooth)
        test_tooth_crop = test_tooth.crop_wrt_wear(reference_tooth)
        if verbose:
            draw_meshes(ref_tooth_crop, test_tooth_crop)

        result = ransac_icp(ref_tooth_crop, test_tooth_crop)
        reference_tooth = reference_tooth.transform(result.transformation)
        if verbose:
            draw_result(reference_tooth, test_tooth)

        colored_tooth, wear = draw_heatmap(reference_tooth, test_tooth, return_max=True)
        tooth_pairs[fdi] = colored_tooth
        wears[fdi] = wear



        # wear_idx, wear_mm, volume_mm3 = reference_tooth.measure_wear(test_tooth, normal)
        # print(f'Maximum profile loss: {wear_mm:.3f}mm')
        # print(f'Overall volume loss: {volume_mm3:.3f}mm3')
        # draw_heatmap(reference_tooth, test_tooth, verbose=True)
        # draw_result(reference_tooth, test_tooth, wear_idx, normal)

    return tooth_pairs, wears


def patient_heights(
    root: Path, patient: str,
    verbose: bool=False,
):
    files = sorted(root.glob(f'{patient}*'))
    times = [f.name.split('_')[1] for f in files]
    followups = set([time for time in times if time[0] != '0'])

    df = pd.DataFrame({'FDI': fdis})
    for followup in sorted(followups):
        wears = {}
        for arch in ['mandible', 'maxilla']:
            tooth_pairs, arch_wears = main(
                root=root.parent,
                reference_stem=f'{patient}_{times[0]}_{arch}',
                test_stem=f'{patient}_{followup}_{arch}',
                verbose=False,
            )

            if verbose:
                draw_meshes(*list(tooth_pairs.values()))
            wears.update(arch_wears)
        
        # save most negative distance of each tooth
        df[f'{times[0]}-{followup}'] = [wears.get(fdi, {'max': np.nan})['max'] for fdi in fdis]

    return df


if __name__ == '__main__':
    root = Path('methodology study')
    verbose = False

    dfs = {}
    # for patient in ['A20', 'A24', 'A29', 'A25', 'A28', 'A46', 'A41', 'A-27']:
    for patient in ['A-17', 'A-26']:
        print('Patient', patient)
        df = patient_heights(root / 'AI labels', patient, verbose)

        patient = f'{patient[0]}-{patient[1:]}' if '-' not in patient else patient
        dfs[patient] = df

    with pd.ExcelWriter(root / 'ours_heights_intra.xlsx') as writer:
        for patient, df in dfs.items():
            df.to_excel(writer, sheet_name=patient, index=False)
