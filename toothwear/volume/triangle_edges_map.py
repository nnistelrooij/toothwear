from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


def update_test_triangle_edges_map(
    out: Dict[
        int,
        Tuple[NDArray[np.int64], NDArray[np.float64]]
    ],
    intersections: List[NDArray[np.float64]],
    vertex_idxs_list: List[NDArray[np.int64]],
    test_triangle_idxs: List[NDArray[np.int64]],
) -> Dict[
    int,
    Tuple[NDArray[np.int64], NDArray[np.float32]],
]:
    for triangle_idxs, inters, vertex_idxs in zip(
        test_triangle_idxs, intersections, vertex_idxs_list,
    ):
        inters = np.stack((inters[:-1], inters[1:])).transpose(1, 0, 2)
        vertex_idxs = np.stack((vertex_idxs[:-1], vertex_idxs[1:])).T
        for key, inters, vertex_idxs in zip(triangle_idxs, inters, vertex_idxs):
            if key not in out:
                out[key] = (np.ones(0, dtype=int), np.ones((0, 3)))

            if key == 471:
                k = 3

            out[key] = (
                np.concatenate((out[key][0], vertex_idxs)),
                np.concatenate((out[key][1], inters)),
            )

    assert test_triangle_idxs[0][0] == test_triangle_idxs[-1][-1]
    key_copy = key
    
    over_idx = vertex_idxs_list[-1][-1]
    for key in out:
        out[key][0][out[key][0] == over_idx] = vertex_idxs_list[0][0]

    prev_key = triangle_idxs[-2]
    prev_triangle_idx = out[prev_key][0][-1]

    prev_triangle_mask = out[key_copy][0] == prev_triangle_idx
    current_start_idx = prev_triangle_mask.shape[0] - 1 - prev_triangle_mask[::-1].argmax()
    current_start_idx = 2 * int(current_start_idx // 2)  # make it even

    out[key_copy] = tuple([
        np.concatenate((
            out[key_copy][i][current_start_idx:],
            out[key_copy][i][:current_start_idx],
        )) for i in range(2)
    ])

    return out