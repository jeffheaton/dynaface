import os
from typing import Tuple

import cv2
import numpy as np
import pkg_resources

# My libs
from dynaface.spiga.data.loaders.augmentors.utils import rotation_matrix_to_euler

# Model file nomenclature
model_file_dft = (
    pkg_resources.resource_filename("dynaface.spiga", "data/models3D")
    + "/mean_face_3D_{num_ldm}.txt"
)


class PositPose:

    def __init__(
        self,
        ldm_ids,
        focal_ratio=1,
        selected_ids=None,
        max_iter=100,
        fix_bbox=True,
        model_file=model_file_dft,
    ):

        # Load 3D face model
        model3d_world, model3d_ids = self._load_world_shape(ldm_ids, model_file)

        # Generate id mask to pick only the robust landmarks for posit
        if selected_ids is None:
            model3d_mask = np.ones(len(ldm_ids))
        else:
            model3d_mask = np.zeros(len(ldm_ids))
            for index, posit_id in enumerate(model3d_ids):
                if posit_id in selected_ids:
                    model3d_mask[index] = 1

        self.ldm_ids = ldm_ids  # Ids from the database
        self.model3d_world = model3d_world  # Model data
        self.model3d_ids = model3d_ids  # Model ids
        self.model3d_mask = model3d_mask  # Model mask ids
        self.max_iter = max_iter  # Refinement iterations
        self.focal_ratio = focal_ratio  # Camera matrix focal length ratio
        self.fix_bbox = (
            fix_bbox  # Camera matrix centered on image (False to centered on bbox)
        )

    def __call__(self, sample):

        landmarks = sample["landmarks"]
        mask = sample["mask_ldm"]

        # Camera matrix
        img_shape = np.array(sample["image"].shape)[0:2]
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
            rot_matrix, trl_matrix = np.eye(3, dtype=float), np.array([0, 0, 0])
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

    def _load_world_shape(self, ldm_ids, model_file):
        return load_world_shape(ldm_ids, model_file=model_file)

    def _camera_matrix(self, bbox):
        focal_length_x = bbox[2] * self.focal_ratio
        focal_length_y = bbox[3] * self.focal_ratio
        face_center = (bbox[0] + (bbox[2] * 0.5)), (bbox[1] + (bbox[3] * 0.5))

        cam_matrix = np.array(
            [
                [focal_length_x, 0, face_center[0]],
                [0, focal_length_y, face_center[1]],
                [0, 0, 1],
            ]
        )
        return cam_matrix

    def _set_correspondences(self, landmarks, mask):
        # Correspondences using labelled and robust landmarks
        img_mask = np.logical_and(mask, self.model3d_mask)
        img_mask = img_mask.astype(bool)

        image_pts = landmarks[img_mask]
        world_pts = self.model3d_world[img_mask]
        return world_pts, image_pts

    def _modern_posit(self, world_pts, image_pts, cam_matrix):
        return modern_posit(world_pts, image_pts, cam_matrix, self.max_iter)

    def _project_points(self, rot, trl, cam_matrix, norm=None):
        # Perspective projection model
        trl = np.expand_dims(trl, 1)
        extrinsics = np.concatenate((rot, trl), 1)
        proj_matrix = np.matmul(cam_matrix, extrinsics)

        # Homogeneous landmarks
        pts = self.model3d_world
        ones = np.ones(pts.shape[0])
        ones = np.expand_dims(ones, 1)
        pts_hom = np.concatenate((pts, ones), 1)

        # Project landmarks
        pts_proj = np.matmul(proj_matrix, pts_hom.T).T
        pts_proj = pts_proj / np.expand_dims(pts_proj[:, 2], 1)  # Lambda = 1

        if norm is not None:
            pts_proj[:, 0] /= norm[0]
            pts_proj[:, 1] /= norm[1]
        return pts_proj[:, :-1]


def load_world_shape(db_landmarks, model_file=model_file_dft):

    # Load 3D mean face coordinates
    num_ldm = len(db_landmarks)
    filename = model_file.format(num_ldm=num_ldm)
    if not os.path.exists(filename):
        raise ValueError("No 3D model find for %i landmarks" % num_ldm)

    posit_landmarks = np.genfromtxt(
        filename, delimiter="|", dtype=int, usecols=0
    ).tolist()
    mean_face_3D = np.genfromtxt(
        filename, delimiter="|", dtype=(float, float, float), usecols=(1, 2, 3)
    ).tolist()
    world_all = len(mean_face_3D) * [None]
    index_all = len(mean_face_3D) * [None]

    for cont, elem in enumerate(mean_face_3D):
        pt3d = [elem[2], -elem[0], -elem[1]]
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

    # Number of landmarks and homogeneous world points
    num_landmarks: int = image_pts.shape[0]
    ones_column: np.ndarray = np.ones((num_landmarks, 1))
    homogeneous_world_pts: np.ndarray = np.concatenate((world_pts, ones_column), axis=1)

    # Compute pseudo-inverse of homogeneous world points
    pseudo_inverse: np.ndarray = np.linalg.pinv(homogeneous_world_pts)

    # Normalize image points based on camera matrix
    focal_length: float = cam_matrix[0, 0]
    img_center_x: float = cam_matrix[0, 2]
    img_center_y: float = cam_matrix[1, 2]
    centered_pts: np.ndarray = np.zeros((num_landmarks, 2))
    centered_pts[:, 0] = (image_pts[:, 0] - img_center_x) / focal_length
    centered_pts[:, 1] = (image_pts[:, 1] - img_center_y) / focal_length

    # Initial projections for POSIT loop
    proj_x: np.ndarray = centered_pts[:, 0]
    proj_y: np.ndarray = centered_pts[:, 1]

    # Initialize translation and rotation vectors
    trans_x, trans_y, trans_z = 0.0, 0.0, 0.0
    rot_vec1, rot_vec2, rot_vec3 = (
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
    )

    # POSIT iteration loop
    for iteration in range(max_iters):
        # Project 3D points to normalized image space
        proj_x_world: np.ndarray = np.dot(pseudo_inverse, proj_x)
        proj_y_world: np.ndarray = np.dot(pseudo_inverse, proj_y)

        # Estimate translation and scale
        norm_proj_x: float = 1.0 / np.linalg.norm(proj_x_world[:3])
        norm_proj_y: float = 1.0 / np.linalg.norm(proj_y_world[:3])
        trans_z: float = np.sqrt(norm_proj_x * norm_proj_y)

        # Scale and normalize rotation vectors
        rot_vec1_norm: np.ndarray = proj_x_world * trans_z
        rot_vec2_norm: np.ndarray = proj_y_world * trans_z
        rot_vec1 = np.clip(rot_vec1_norm[:3], -1, 1)
        rot_vec2 = np.clip(rot_vec2_norm[:3], -1, 1)
        rot_vec3 = np.cross(rot_vec1, rot_vec2)

        # Update translation values
        trans_x, trans_y = rot_vec1_norm[3], rot_vec2_norm[3]
        rot_vec3_with_trans_z: np.ndarray = np.concatenate((rot_vec3, [trans_z]))

        # Compute epsilon and update projections
        epsilon: np.ndarray = (
            np.dot(homogeneous_world_pts, rot_vec3_with_trans_z) / trans_z
        )
        prev_proj_x, prev_proj_y = proj_x, proj_y
        proj_x = epsilon * centered_pts[:, 0]
        proj_y = epsilon * centered_pts[:, 1]

        # Check for convergence
        delta_proj_x: np.ndarray = proj_x - prev_proj_x
        delta_proj_y: np.ndarray = proj_y - prev_proj_y
        delta: float = focal_length**2 * (
            np.dot(delta_proj_x.T, delta_proj_x) + np.dot(delta_proj_y.T, delta_proj_y)
        )

        if iteration > 0 and delta < 0.01:  # Converged
            break

    # Create rotation matrix and translation vector
    rot_matrix: np.ndarray = np.array(
        [rot_vec1, rot_vec2, rot_vec3]
    ).T  # Transpose for correct orientation
    trl_matrix: np.ndarray = np.array([trans_x, trans_y, trans_z])

    # Convert to nearest orthogonal rotation matrix using SVD
    _, u_matrix, vt_matrix = cv2.SVDecomp(rot_matrix)
    rot_matrix = np.matmul(u_matrix, vt_matrix)

    return rot_matrix, trl_matrix
