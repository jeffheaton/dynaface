import dynaface.spiga.models.gnn.pose_proj as pproj
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynaface.spiga.models.cnn.cnn_multitask import MultitaskCNN
from dynaface.spiga.models.gnn.step_regressor import (
    RelativePositionEncoder,
    StepRegressor,
)
from typing import Any, Dict, Tuple


class SPIGA(nn.Module):
    """
    SPIGA model for landmark regression using cascaded graph attention networks (GATs)
    and convolutional neural networks (CNNs).

    Attributes:
        steps (int): Number of cascaded regressors.
        embedded_dim (int): Dimensionality of the embedded features.
        nstack (int): Number of stacked GAT layers per step.
        kwindow (int): Kernel window size for cropping.
        swindow (float): Scale factor for the cropped window at the first step.
        offset_ratio (List[float]): Offset ratios for updating coordinates at each step.
        num_landmarks (int): Number of landmarks.
        num_edges (int): Number of edges.
        visual_cnn (MultitaskCNN): CNN backbone for feature extraction.
        img_res (int): Input image resolution.
        visual_res (int): Resolution of the output feature map from the CNN.
        visual_dim (int): Number of channels in the CNN's output feature map.
        channels_pose (int): Number of channels for head pose estimation.
        pose_fc (nn.Linear): Fully connected layer for head pose estimation.
        shape_encoder (nn.ModuleList): List of relative positional encoder modules.
        diagonal_mask (nn.Parameter): Mask to exclude self-distances in pairwise computations.
        theta_S (nn.ParameterList): List of affine transformation scale parameters per step.
        conv_window (nn.ModuleList): List of convolutional layers for visual feature extraction.
        gcn (nn.ModuleList): List of GAT-based step regressors.
    """

    def __init__(
        self,
        num_landmarks: int = 98,
        num_edges: int = 15,
        steps: int = 3,
        **kwargs: Any
    ) -> None:
        """
        Initializes the SPIGA model.

        Args:
            num_landmarks (int): Number of landmarks (default: 98).
            num_edges (int): Number of edges (default: 15).
            steps (int): Number of cascaded regressors (default: 3).
            **kwargs: Additional keyword arguments.
        """
        super(SPIGA, self).__init__()

        # Model parameters
        self.steps: int = steps  # Cascaded regressors
        self.embedded_dim: int = 512  # GAT input channel
        self.nstack: int = 4  # Number of stacked GATs per step
        self.kwindow: int = 7  # Output cropped window dimension (kernel)
        self.swindow: float = (
            0.25  # Scale of the cropped window at first step (Default 25% of input feature map)
        )
        self.offset_ratio = [self.swindow / (2**step) / 2 for step in range(self.steps)]

        # CNN parameters
        self.num_landmarks: int = num_landmarks
        self.num_edges: int = num_edges

        # Initialize backbone
        self.visual_cnn: MultitaskCNN = MultitaskCNN(
            num_landmarks=self.num_landmarks, num_edges=self.num_edges
        )
        # Features dimensions
        self.img_res: int = self.visual_cnn.img_res
        self.visual_res: int = self.visual_cnn.out_res
        self.visual_dim: int = self.visual_cnn.ch_dim

        # Initialize Pose head
        self.channels_pose: int = 6
        self.pose_fc: nn.Linear = nn.Linear(self.visual_cnn.ch_dim, self.channels_pose)

        # Initialize feature extractors:
        # Relative positional encoder
        shape_dim: int = 2 * (self.num_landmarks - 1)
        shape_encoder = []
        for step in range(self.steps):
            shape_encoder.append(
                RelativePositionEncoder(shape_dim, self.embedded_dim, [256, 256])
            )
        self.shape_encoder: nn.ModuleList = nn.ModuleList(shape_encoder)
        # Diagonal mask used to compute relative positions
        diagonal_mask = (
            torch.ones(self.num_landmarks, self.num_landmarks)
            - torch.eye(self.num_landmarks)
        ).type(torch.bool)
        self.diagonal_mask: nn.Parameter = nn.parameter.Parameter(
            diagonal_mask, requires_grad=False
        )

        # Visual feature extractor
        conv_window = []
        theta_S = []
        for step in range(self.steps):
            # S matrix per step
            WH: int = self.visual_res  # Width/height of feature map
            Wout: float = self.swindow / (2**step) * WH  # Width/height of the window
            K: int = self.kwindow  # Kernel or resolution of the window
            scale: float = (
                K / WH * (Wout - 1) / (K - 1)
            )  # Scale for the affine transformation
            # Rescale matrix S
            theta_S_stp = torch.tensor([[scale, 0], [0, scale]])
            theta_S.append(nn.parameter.Parameter(theta_S_stp, requires_grad=False))

            # Convolutional to embedded to BxLxCx1x1
            conv_window.append(
                nn.Conv2d(self.visual_dim, self.embedded_dim, self.kwindow)
            )

        self.theta_S: nn.ParameterList = nn.ParameterList(theta_S)
        self.conv_window: nn.ModuleList = nn.ModuleList(conv_window)

        # Initialize GAT modules
        self.gcn: nn.ModuleList = nn.ModuleList(
            [
                StepRegressor(self.embedded_dim, 256, self.nstack)
                for i in range(self.steps)
            ]
        )

    def forward(
        self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Performs the forward pass of the SPIGA model.

        Args:
            data (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                A tuple containing the input image, 3D model, and camera matrix.

        Returns:
            Dict[str, Any]: A dictionary containing updated landmark projections, GAT probabilities,
                            pose estimation, and other intermediate features.
        """
        # Process backbone to obtain initial projections and features
        pts_proj, features = self.backbone_forward(data)
        visual_field: torch.Tensor = features["VisualField"][-1]

        # Iterate over each regression step
        gat_prob: list[torch.Tensor] = []
        features["Landmarks"] = []
        for step in range(self.steps):
            # Generate embedded features from visual and shape information
            embedded_ft = self.extract_embedded(pts_proj, visual_field, step)

            # GAT inference
            offset, gat_prob = self.gcn[step](embedded_ft, gat_prob)
            offset = F.hardtanh(offset)

            # Update landmark projections
            pts_proj = pts_proj + self.offset_ratio[step] * offset
            features["Landmarks"].append(pts_proj.clone())

        features["GATProb"] = gat_prob
        return features

    def backbone_forward(
        self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Processes the input data through the CNN backbone and projects the 3D model.

        Args:
            data (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                A tuple containing the input image, 3D model, and camera matrix.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: A tuple where the first element is the projected
                landmark points and the second element is a dictionary of features from the CNN.
        """
        imgs, model3d, cam_matrix = data

        # HourGlass forward pass
        features: Dict[str, Any] = self.visual_cnn(imgs)

        # Head pose estimation
        pose_raw: torch.Tensor = features["HGcore"][-1]
        B, L, _, _ = pose_raw.shape
        pose: torch.Tensor = pose_raw.reshape(B, L)
        pose = self.pose_fc(pose)
        features["Pose"] = pose.clone()

        # Project 3D model points onto the image plane
        euler: torch.Tensor = pose[:, 0:3]
        trl: torch.Tensor = pose[:, 3:]
        rot: torch.Tensor = pproj.euler_to_rotation_matrix(euler)
        pts_proj: torch.Tensor = pproj.projectPoints(model3d, rot, trl, cam_matrix)
        pts_proj = pts_proj / self.visual_res

        return pts_proj, features

    def extract_embedded(
        self, pts_proj: torch.Tensor, receptive_field: torch.Tensor, step: int
    ) -> torch.Tensor:
        """
        Extracts and combines visual and shape features for embedding.

        Args:
            pts_proj (torch.Tensor): Projected landmark points of shape (B, L, 2).
            receptive_field (torch.Tensor): Visual feature map from the CNN.
            step (int): Current regression step.

        Returns:
            torch.Tensor: The embedded features for the current step.
        """
        # Extract visual features
        visual_ft: torch.Tensor = self.extract_visual_embedded(
            pts_proj, receptive_field, step
        )
        # Extract shape features via pairwise distances
        shape_ft: torch.Tensor = self.calculate_distances(pts_proj)
        shape_ft = self.shape_encoder[step](shape_ft)
        # Combine visual and shape features
        embedded_ft: torch.Tensor = visual_ft + shape_ft
        return embedded_ft

    def extract_visual_embedded(
        self, pts_proj: torch.Tensor, receptive_field: torch.Tensor, step: int
    ) -> torch.Tensor:
        """
        Extracts visual features using an affine transformation and convolution.

        Args:
            pts_proj (torch.Tensor): Projected landmark points of shape (B, L, 2) with values in [0, 1].
            receptive_field (torch.Tensor): Visual feature map from the CNN.
            step (int): Current regression step.

        Returns:
            torch.Tensor: Extracted visual features.
        """
        # Generate affine transformation matrices
        B, L, _ = pts_proj.shape  # (B, L, 2)
        centers: torch.Tensor = (
            pts_proj + 0.5 / self.visual_res
        )  # Adjust centers, shape: (B, L, 2)
        centers = centers.reshape(B * L, 2)
        theta_trl: torch.Tensor = (-1 + centers * 2).unsqueeze(-1)  # (B*L, 2, 1)
        theta_s: torch.Tensor = self.theta_S[step]  # (2, 2)
        theta_s = theta_s.repeat(B * L, 1, 1)  # (B*L, 2, 2)
        theta: torch.Tensor = torch.cat((theta_s, theta_trl), -1)  # (B*L, 2, 3)

        # Generate crop grid using affine transformation
        B, C, _, _ = receptive_field.shape
        grid = torch.nn.functional.affine_grid(
            theta, (B * L, C, self.kwindow, self.kwindow), align_corners=False
        )
        grid = grid.reshape(B, L, self.kwindow, self.kwindow, 2)
        grid = grid.reshape(B, L, self.kwindow * self.kwindow, 2)

        # Sample the receptive field using the generated grid
        crops = torch.nn.functional.grid_sample(
            receptive_field, grid, padding_mode="zeros", align_corners=False
        )
        crops = crops.transpose(1, 2)  # Shape: (B, L, C, K*K)
        crops = crops.reshape(B * L, C, self.kwindow, self.kwindow)

        # Apply convolution to extract visual features
        visual_ft: torch.Tensor = self.conv_window[step](crops)
        _, Cout, _, _ = visual_ft.shape
        visual_ft = visual_ft.reshape(B, L, Cout)

        return visual_ft

    def calculate_distances(self, pts_proj: torch.Tensor) -> torch.Tensor:
        """
        Calculates pairwise differences between landmark points, excluding self-comparisons.

        Args:
            pts_proj (torch.Tensor): Projected landmark points of shape (B, L, 2).

        Returns:
            torch.Tensor: Pairwise differences with self-distances removed, reshaped to (B, L, -1).
        """
        B, L, _ = pts_proj.shape  # (B, L, 2)
        pts_a: torch.Tensor = pts_proj.unsqueeze(-2).repeat(1, 1, L, 1)
        pts_b: torch.Tensor = pts_a.transpose(1, 2)
        dist: torch.Tensor = pts_a - pts_b
        dist_wo_self: torch.Tensor = dist[:, self.diagonal_mask, :].reshape(B, L, -1)
        return dist_wo_self
