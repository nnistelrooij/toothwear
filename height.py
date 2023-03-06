from pathlib import Path

from toothwear.io import load_dental_mesh
from toothwear.register import (
    procrustes,
    ransac_icp,
)
from toothwear.visualization import (
    draw_heatmap,
    draw_landmark,
    draw_meshes,
    draw_result,
)


def main(
    root: Path,
    reference_stem: str,
    test_stem: str,
):
    reference = load_dental_mesh(root, reference_stem, reference=True)
    test = load_dental_mesh(root, test_stem, reference=False)
    # draw_meshes([reference, test], color=False)
    draw_meshes([reference, test])

    T = procrustes(reference, test)
    reference = reference.transform(T)
    normal = reference[reference.labels > 0].normals.mean(axis=0)
    # draw_meshes([reference, test])

    reference_teeth = reference.crop_teeth()
    test_teeth = test.crop_teeth()

    keys = set(reference_teeth) & set(test_teeth)
    for label in sorted(keys):
        print('label:', label)
        reference_tooth = reference_teeth[label]
        test_tooth = test_teeth[label]
        # draw_meshes([reference_tooth, test_tooth])

        reference_tooth_crop = reference_tooth.crop_wrt_normal(-normal)
        test_tooth_crop = test_tooth.crop_wrt_normal(-normal)
        # draw_meshes([reference_tooth_crop, test_tooth_crop])

        T = ransac_icp(reference_tooth_crop, test_tooth_crop)
        reference_tooth = reference_tooth.transform(T)
        normal = normal @ T[:3, :3]
        # draw_meshes([reference_tooth, test_tooth])
        # draw_heatmap(reference_tooth, test_tooth)

        reference_tooth_crop = reference_tooth.crop_wrt_wear(test_tooth)
        test_tooth_crop = test_tooth.crop_wrt_wear(reference_tooth)
        # draw_meshes([reference_tooth_crop, test_tooth_crop])

        T = ransac_icp(reference_tooth_crop, test_tooth_crop)
        reference_tooth = reference_tooth.transform(T)
        # draw_meshes([reference_tooth, test_tooth])
        # draw_heatmap(reference_tooth, test_tooth)

        wear_idx, wear_mm = reference_tooth.measure_wear(test_tooth, normal)
        print(f'Wear: {wear_mm:.3f}mm')
        draw_result(reference_tooth, test_tooth, wear_idx, normal)


if __name__ == '__main__':
    root = Path('/home/mkaailab/Documents/toothwear2')
    main(
        root=root,
        reference_stem='A19_2013_maxilla',
        test_stem='A19_2020_maxilla',
    )
    main(
        root=root,
        reference_stem='A19_2013_mandible',
        test_stem='A19_2020_mandible',
    )
    # main(
    #     root=root,
    #     reference_stem='A21_2012_maxilla',
    #     test_stem='A21_2013_maxilla',
    # )
    main(
        root=root,
        reference_stem='A21_2012_mandible',
        test_stem='A21_2013_mandible',
    )
    # main(
    #     root=root,
    #     reference_stem='2013 maxilla',
    #     test_stem='2021 maxilla',
    # )
    # main(
    #     root=root,
    #     reference_stem='2013 mandible',
    #     test_stem='2021 mandible',
    # )
