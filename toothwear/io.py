from itertools import chain
from pathlib import Path
from typing import List

from toothwear.teeth import DentalMesh


def load_dental_mesh(
    root: Path,
    stem: str,
    reference: bool,
    extensions: List[str]=['obj', 'stl'],
) -> DentalMesh:
    mesh_file = next(chain(*[root.glob(f'**/{stem}*.{e}') for e in extensions]))
    ann_file = next(root.glob(f'**/{stem}*.json'))

    mesh = DentalMesh.from_files(
        mesh_file, ann_file, reference=reference,
    )

    return mesh
