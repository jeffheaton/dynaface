from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple

from dynaface.spiga.data.loaders.dl_config import DatabaseStruct

MODELS_URL: Dict[str, str] = {
    "wflw": "https://drive.google.com/uc?export=download&confirm=yes&id=1h0qA5ysKorpeDNRXe9oYkVcVe8UYyzP7",
    "300wpublic": "https://drive.google.com/uc?export=download&confirm=yes&id=1YrbScfMzrAAWMJQYgxdLZ9l57nmTdpQC",
    "300wprivate": "https://drive.google.com/uc?export=download&confirm=yes&id=1fYv-Ie7n14eTD0ROxJYcn6SXZY5QU9SM",
    "merlrav": "https://drive.google.com/uc?export=download&confirm=yes&id=1GKS1x0tpsTVivPZUk_yrSiMhwEAcAkg6",
    "cofw68": "https://drive.google.com/uc?export=download&confirm=yes&id=1fYv-Ie7n14eTD0ROxJYcn6SXZY5QU9SM",
}


class ModelConfig:
    """
    Configuration class for managing model settings and parameters.

    Attributes:
        model_weights (Optional[str]): The filename for the model weights.
        model_weights_path (Optional[str]): The file path for the model weights.
        load_model_url (bool): Flag to determine if the model weights URL should be loaded.
        model_weights_url (Optional[str]): The URL to download the model weights.
        focal_ratio (float): The camera matrix focal length ratio.
        target_dist (float): The target distance zoom factor around the face.
        image_size (Tuple[int, int]): The size (width, height) of the input image.
        ftmap_size (Tuple[int, int]): The size (width, height) of the output feature map.
        dataset (Optional[DatabaseStruct]): The dataset configuration.
    """

    def __init__(
        self, dataset_name: Optional[str] = None, load_model_url: bool = True
    ) -> None:
        """
        Initializes the ModelConfig instance with default values and updates with the dataset configuration if provided.

        Args:
            dataset_name (Optional[str]): The name of the dataset to load configuration for.
            load_model_url (bool): Flag indicating whether to load the model URL.
        """
        # Model configuration
        self.model_weights: Optional[str] = None
        self.model_weights_path: Optional[str] = None
        self.load_model_url: bool = load_model_url
        self.model_weights_url: Optional[str] = None

        # Pretreatment parameters
        self.focal_ratio: float = 1.5  # Camera matrix focal length ratio.
        self.target_dist: float = 1.6  # Target distance zoom in/out around face.
        self.image_size: Tuple[int, int] = (256, 256)

        # Output configuration
        self.ftmap_size: Tuple[int, int] = (64, 64)

        # Dataset configuration
        self.dataset: Optional[DatabaseStruct] = None

        if dataset_name is not None:
            self.update_with_dataset(dataset_name)

    def update_with_dataset(self, dataset_name: str) -> None:
        """
        Updates the configuration based on the specified dataset.

        This method creates a configuration dictionary using the dataset name,
        sets the appropriate model weights filename, and updates the configuration.

        Args:
            dataset_name (str): The name of the dataset to use for updating the configuration.
        """
        config_dict: Dict[str, Any] = {
            "dataset": DatabaseStruct(dataset_name),
            "model_weights": f"spiga_{dataset_name}.pt",
        }

        if dataset_name == "cofw68":  # Test only
            config_dict["model_weights"] = "spiga_300wprivate.pt"

        if self.load_model_url:
            config_dict["model_weights_url"] = MODELS_URL[dataset_name]

        self.update(config_dict)

    def update(self, params_dict: Dict[str, Any]) -> None:
        """
        Updates the instance attributes based on the provided dictionary.

        Args:
            params_dict (Dict[str, Any]): A dictionary containing configuration keys and their corresponding values.

        Raises:
            Warning: If an unknown configuration option is provided.
        """
        state_dict = self.state_dict()
        for k, v in params_dict.items():
            if k in state_dict or hasattr(self, k):
                setattr(self, k, v)
            else:
                raise Warning(f"Unknown option: {k}: {v}")

    def state_dict(self) -> "OrderedDict[str, Any]":
        """
        Returns an ordered dictionary containing the current state of the configuration.

        Only attributes that do not start with an underscore are included.

        Returns:
            OrderedDict[str, Any]: An ordered dictionary of configuration attributes.
        """
        state_dict: "OrderedDict[str, Any]" = OrderedDict()
        for k in self.__dict__.keys():
            if not k.startswith("_"):
                state_dict[k] = getattr(self, k)
        return state_dict
