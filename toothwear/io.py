from pathlib import Path

from toothwear.teeth import DentalMesh


def load_dental_mesh(
    root: Path,
    stem: str,
    reference: bool,
) -> DentalMesh:
    mesh_file = next(root.glob(f'**/{stem}.obj'))
    ann_file = next(root.glob(f'**/{stem}.json'))

    mesh = DentalMesh.from_files(
        mesh_file, ann_file, reference=reference,
    )

    return mesh
