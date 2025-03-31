import json
import os
from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import pkg_resources

db_img_path = pkg_resources.resource_filename("dynaface", "spiga/data/databases")
db_anns_path = (
    pkg_resources.resource_filename("dynaface", "spiga/data/annotations")
    + "/{database}/{file_name}.json"
)


class DatabaseStruct:
    """
    Handles database-specific information for facial landmark analysis.

    Attributes:
        name (str): Name of the database.
        ldm_ids (List[int]): List of landmark IDs.
        ldm_flip_order (List[int]): Landmark order for horizontal flip.
        ldm_edges_matrix (List[List[int]]): Landmark edges matrix.
        num_landmarks (int): Number of landmarks.
        num_edges (int): Number of landmark edges.
        fields (List[str]): List of field names for the database.
    """

    def __init__(self, database_name: str) -> None:
        """
        Initializes the DatabaseStruct with the given database name.

        Args:
            database_name (str): Name of the database to load.

        Raises:
            ValueError: If database-specific information is missing.
        """

        self.name = database_name
        self.ldm_ids, self.ldm_flip_order, self.ldm_edges_matrix = (
            self._get_database_specifics()
        )
        self.num_landmarks = len(self.ldm_ids)
        self.num_edges = len(self.ldm_edges_matrix[0]) - 1
        self.fields = ["imgpath", "bbox", "headpose", "ids", "landmarks", "visible"]

    def _get_database_specifics(
        self,
    ) -> Tuple[List[int], List[int], Optional[List[List[int]]]]:
        """
        Retrieves database-specific information such as landmark IDs,
        flip order, and edge matrix.

        Returns:
            Tuple[List[int], List[int], Optional[List[List[int]]]]:
            - Landmark IDs
            - Landmark flip order
            - Landmark edges matrix (if available)

        Raises:
            ValueError: If the database-specific information file is missing.
        """

        database_name = self.name
        db_info_file = db_anns_path.format(database=database_name, file_name="db_info")
        ldm_edges_matrix = None

        if os.path.exists(db_info_file):
            with open(db_info_file) as jsonfile:
                db_info = json.load(jsonfile)

            ldm_ids = db_info["ldm_ids"]
            ldm_flip_order = db_info["ldm_flip_order"]
            if "ldm_edges_matrix" in db_info.keys():
                ldm_edges_matrix = db_info["ldm_edges_matrix"]

        else:
            print(db_anns_path)
            print(db_info_file)
            raise ValueError(
                "Database "
                + database_name
                + "specifics not defined. Missing db_info.json"
            )

        return ldm_ids, ldm_flip_order, ldm_edges_matrix

    def state_dict(self) -> OrderedDict[str, Any]:
        """
        Creates a dictionary representing the current state of the object.

        Returns:
            OrderedDict[str, Any]: Dictionary with object attributes and their values.
        """
        state_dict = OrderedDict()
        for k in self.__dict__.keys():
            if not k.startswith("_"):
                state_dict[k] = getattr(self, k)

        return state_dict

    def __str__(self) -> str:
        """
        Returns a string representation of the database information.

        Returns:
            str: Formatted string containing database-specific information.
        """
        state_dict = self.state_dict()
        text = "Database {\n"
        for k, v in state_dict.items():
            text += "\t{}: {}\n".format(k, v)
        text += "\t}\n"
        return text
