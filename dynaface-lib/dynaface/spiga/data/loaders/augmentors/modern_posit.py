import os
from typing import Any, List, Optional, Tuple

import numpy as np
import pkg_resources
from numpy.typing import NDArray

# Model file nomenclature
model_file_dft: str = (
    pkg_resources.resource_filename("dynaface.spiga", "data/models3D")
    + "/mean_face_3D_{num_ldm}.txt"
)


class PositPose:
    def __init__(
        self,
        ldm_ids: List[int],
        focal_ratio: float = 1.0,
        selected_ids: Optional[List[int]] = None,
        max_iter: int = 100,
        fix_bbox: bool = True,
        model_file: str = model_file_dft,
    ) -> None:
        # Load 3D face model
        model3d_world, model3d_ids = self._load_world_shape(ldm_ids, model_file)

        # Generate id mask to pick only the robust landmarks for posit
        if selected_ids is None:
            model3d_mask = np.ones(len(ldm_ids), dtype=bool)
        else:
            model3d_mask = np.zeros(len(ldm_ids), dtype=bool)
            for index, posit_id in enumerate(model3d_ids):
                if posit_id in selected_ids:
                    model3d_mask[index] = True

        self.ldm_ids: List[int] = ldm_ids  # Ids from the database
        self.model3d_world: NDArray[Any] = model3d_world  # Model data
        self.model3d_ids: NDArray[Any] = model3d_ids  # Model ids
        self.model3d_mask: NDArray[Any] = model3d_mask  # Model mask ids
        self.max_iter: int = max_iter  # Refinement iterations
        self.focal_ratio: float = focal_ratio  # Camera matrix focal length ratio
        self.fix_bbox: bool = fix_bbox  # Camera matrix centered on image

    def _load_world_shape(
        self, ldm_ids: List[int], model_file: str
    ) -> Tuple[NDArray[Any], NDArray[Any]]:
        return load_world_shape(ldm_ids, model_file=model_file)

    def _camera_matrix(self, bbox: List[float]) -> NDArray[Any]:
        focal_length_x: float = bbox[2] * self.focal_ratio
        focal_length_y: float = bbox[3] * self.focal_ratio
        face_center: Tuple[float, float] = (
            bbox[0] + (bbox[2] * 0.5),
            bbox[1] + (bbox[3] * 0.5),
        )

        cam_matrix: NDArray[Any] = np.array(
            [
                [focal_length_x, 0, face_center[0]],
                [0, focal_length_y, face_center[1]],
                [0, 0, 1],
            ]
        )
        return cam_matrix


def load_world_shape(
    db_landmarks: List[int], model_file: str = model_file_dft
) -> Tuple[NDArray[Any], NDArray[Any]]:
    # Load 3D mean face coordinates
    num_ldm = len(db_landmarks)
    filename = model_file.format(num_ldm=num_ldm)
    if not os.path.exists(filename):
        raise ValueError("No 3D model found for %i landmarks" % num_ldm)

    posit_landmarks = np.genfromtxt(
        filename, delimiter="|", dtype=np.int32, usecols=[0]
    ).tolist()
    mean_face_3D = np.genfromtxt(
        filename, delimiter="|", dtype=float, usecols=[1, 2, 3]
    ).tolist()
    world_all: List[List[float]] = [[] for _ in range(len(mean_face_3D))]
    index_all: List[int] = [-1] * len(mean_face_3D)  # Use -1 to indicate uninitialized

    for cont, elem in enumerate(mean_face_3D):
        pt3d: List[float] = [elem[2], -elem[0], -elem[1]]
        lnd_idx = db_landmarks.index(posit_landmarks[cont])
        world_all[lnd_idx] = pt3d
        index_all[lnd_idx] = posit_landmarks[cont]

    return np.array(world_all), np.array(index_all)
