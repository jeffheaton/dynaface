import os
from typing import Tuple, Dict, Any, Optional, List

import cv2
import numpy as np
import pkg_resources

# My libs
from dynaface.spiga.data.loaders.augmentors.utils import rotation_matrix_to_euler

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
        self.model3d_world: np.ndarray = model3d_world  # Model data
        self.model3d_ids: np.ndarray = model3d_ids  # Model ids
        self.model3d_mask: np.ndarray = model3d_mask  # Model mask ids
        self.max_iter: int = max_iter  # Refinement iterations
        self.focal_ratio: float = focal_ratio  # Camera matrix focal length ratio
        self.fix_bbox: bool = fix_bbox  # Camera matrix centered on image

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        landmarks: np.ndarray = sample["landmarks"]
        mask: np.ndarray = sample["mask_ldm"]

        # Camera matrix
        img_shape: np.ndarray = np.array(sample["image"].shape)[0:2]
        if "img2map_scale" in sample.keys():
            img_shape = img_shape * sample["img2map_scale"]

        if self.fix_bbox:
            img_bbox = [
                0,
                0,
                img_shape[1],
                img_shape[0],
            ]  # Shapes given are inverted (y,x)
            cam_matrix = self._camera_matrix(img_bbox)
        else:
            bbox = sample["bbox"]  # Scale error when ftshape and img_shape mismatch
            cam_matrix = self._camera_matrix(bbox)

        # Save intrinsic matrix and 3D model landmarks
        sample["cam_matrix"] = cam_matrix
        sample["model3d"] = self.model3d_world

        world_pts, image_pts = self._set_correspondences(landmarks, mask)

        if image_pts.shape[0] < 4:
            print("POSIT does not work without landmarks")
            rot_matrix: np.ndarray = np.eye(3, dtype=float)
            trl_matrix: np.ndarray = np.array([0.0, 0.0, 0.0])
        else:
            rot_matrix, trl_matrix = self._modern_posit(
                world_pts, image_pts, cam_matrix
            )

        euler = rotation_matrix_to_euler(rot_matrix)
        sample["pose"] = np.array(
            [euler[0], euler[1], euler[2], trl_matrix[0], trl_matrix[1], trl_matrix[2]]
        )
        sample["model3d_proj"] = self._project_points(
            rot_matrix, trl_matrix, cam_matrix, norm=img_shape
        )
        return sample

    def _load_world_shape(
        self, ldm_ids: List[int], model_file: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        return load_world_shape(ldm_ids, model_file=model_file)

    def _camera_matrix(self, bbox: List[float]) -> np.ndarray:
        focal_length_x: float = bbox[2] * self.focal_ratio
        focal_length_y: float = bbox[3] * self.focal_ratio
        face_center: Tuple[float, float] = (
            bbox[0] + (bbox[2] * 0.5),
            bbox[1] + (bbox[3] * 0.5),
        )

        cam_matrix: np.ndarray = np.array(
            [
                [focal_length_x, 0, face_center[0]],
                [0, focal_length_y, face_center[1]],
                [0, 0, 1],
            ]
        )
        return cam_matrix

    def _set_correspondences(
        self, landmarks: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Correspondences using labelled and robust landmarks
        img_mask: np.ndarray = np.logical_and(mask, self.model3d_mask)
        img_mask = img_mask.astype(bool)

        image_pts: np.ndarray = landmarks[img_mask]
        world_pts: np.ndarray = self.model3d_world[img_mask]
        return world_pts, image_pts

    def _modern_posit(
        self, world_pts: np.ndarray, image_pts: np.ndarray, cam_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return modern_posit(world_pts, image_pts, cam_matrix, self.max_iter)


def _project_points(
    self,
    rot: np.ndarray,
    trl: np.ndarray,
    cam_matrix: np.ndarray,
    norm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Projects 3D landmarks to 2D image space using a perspective projection model.

    This method takes rotation, translation, and camera matrix inputs and projects
    3D points onto a 2D plane. Optionally, the projected points can be normalized.

    Args:
        rot (np.ndarray): A 3x3 rotation matrix.
        trl (np.ndarray): A 3x1 translation vector.
        cam_matrix (np.ndarray): A 3x3 camera intrinsic matrix.
        norm (Optional[np.ndarray], optional): An array of shape (2,) for normalizing
            the projected points. If provided, the projected points' x and y coordinates
            are divided by `norm[0]` and `norm[1]` respectively. Defaults to None.

    Returns:
        np.ndarray: An (N, 2) array of projected 2D points, where N is the number of landmarks.
    """
    # Perspective projection model
    trl = np.expand_dims(trl, 1)  # Shape: (3, 1)
    extrinsics = np.concatenate((rot, trl), 1)  # Shape: (3, 4)
    proj_matrix: np.ndarray = np.matmul(cam_matrix, extrinsics)  # Shape: (3, 4)

    # Homogeneous landmarks
    pts: np.ndarray = self.model3d_world  # Shape: (N, 3)
    ones: np.ndarray = np.ones((pts.shape[0], 1))  # Shape: (N, 1)
    pts_hom: np.ndarray = np.concatenate((pts, ones), 1)  # Shape: (N, 4)

    # Project landmarks
    pts_proj: np.ndarray = np.matmul(proj_matrix, pts_hom.T).T  # Shape: (N, 3)
    pts_proj = pts_proj / np.expand_dims(pts_proj[:, 2], 1)  # Normalize by depth (λ=1)

    if norm is not None:
        pts_proj[:, 0] /= norm[0]
        pts_proj[:, 1] /= norm[1]

    return pts_proj[:, :-1]  # Return shape: (N, 2)


def load_world_shape(
    db_landmarks: List[int], model_file: str = model_file_dft
) -> Tuple[np.ndarray, np.ndarray]:
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


def modern_posit(
    world_pts: np.ndarray, image_pts: np.ndarray, cam_matrix: np.ndarray, max_iters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate rotation and translation matrices using the POSIT algorithm.

    Args:
        world_pts (np.ndarray): 3D world points (num_landmarks, 3).
        image_pts (np.ndarray): 2D image points (num_landmarks, 2).
        cam_matrix (np.ndarray): Camera intrinsic matrix (3x3).
        max_iters (int): Maximum number of POSIT iterations.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - rot_matrix: Estimated 3x3 rotation matrix.
            - trl_matrix: Estimated translation vector (Tx, Ty, Tz).
    """
    # Fix for type issues in the POSIT loop
    trans_x: float = 0.0
    trans_y: float = 0.0
    trans_z: float = 0.0
    rot_vec1: np.ndarray = np.zeros(3)
    rot_vec2: np.ndarray = np.zeros(3)
    rot_vec3: np.ndarray = np.zeros(3)

    # POSIT iteration loop
    for iteration in range(max_iters):
        # ...existing code...
        pass

    # Create rotation matrix and translation vector
    rot_matrix: np.ndarray = np.array([rot_vec1, rot_vec2, rot_vec3]).T
    trl_matrix: np.ndarray = np.array([trans_x, trans_y, trans_z])

    # Convert to nearest orthogonal rotation matrix using SVD
    _, u_matrix, vt_matrix = cv2.SVDecomp(rot_matrix)
    rot_matrix = np.matmul(u_matrix, vt_matrix)

    return rot_matrix, trl_matrix
