from itertools import chain
from pathlib import Path
from typing import List, Union

import open3d

from toothwear.teeth import DentalMesh


def load_dental_mesh(
    root: Path,
    stem: str,
    reference: bool,
    extensions: List[str]=['obj', 'stl', 'ply'],
) -> DentalMesh:
    mesh_file = next(chain(*[root.glob(f'**/{stem}*.{e}') for e in extensions]))
    ann_file = next(root.glob(f'**/{stem}*.json'))

    mesh = DentalMesh.from_files(
        mesh_file, ann_file, reference=reference,
    )

    return mesh


def save_dental_mesh(
    path: Union[Path, str],
    mesh: Union[DentalMesh, open3d.geometry.TriangleMesh],
) -> None:
    if isinstance(mesh, DentalMesh):
        mesh = mesh.to_open3d_triangle_mesh()

    open3d.io.write_triangle_mesh(str(Path(path).with_suffix('.ply')), mesh)
