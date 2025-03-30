import unittest
from unittest.mock import patch, mock_open
import json
from dynaface.spiga.data.loaders.dl_config import DatabaseStruct


class TestDatabaseStruct(unittest.TestCase):

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_initialization_success(self, mock_file, mock_exists):
        mock_exists.return_value = True
        mock_data = {
            "ldm_ids": [1, 2, 3],
            "ldm_flip_order": [3, 2, 1],
            "ldm_edges_matrix": [[0, 1], [1, 2]],
        }
        mock_file.return_value.read.return_value = json.dumps(mock_data)

        db = DatabaseStruct("test_db")

        self.assertEqual(db.name, "test_db")
        self.assertEqual(db.ldm_ids, [1, 2, 3])
        self.assertEqual(db.ldm_flip_order, [3, 2, 1])
        self.assertEqual(db.ldm_edges_matrix, [[0, 1], [1, 2]])
        self.assertEqual(db.num_landmarks, 3)
        self.assertEqual(db.num_edges, 1)
        self.assertEqual(
            db.fields, ["imgpath", "bbox", "headpose", "ids", "landmarks", "visible"]
        )

    @patch("os.path.exists")
    def test_initialization_failure(self, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(ValueError) as context:
            DatabaseStruct("invalid_db")
        self.assertIn(
            "specifics not defined. Missing db_info.json", str(context.exception)
        )

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_state_dict(self, mock_file, mock_exists):
        mock_exists.return_value = True
        mock_data = {
            "ldm_ids": [1, 2, 3],
            "ldm_flip_order": [3, 2, 1],
            "ldm_edges_matrix": [[0, 1], [1, 2]],
        }
        mock_file.return_value.read.return_value = json.dumps(mock_data)

        db = DatabaseStruct("test_db")
        state_dict = db.state_dict()
        expected_state_dict = {
            "name": "test_db",
            "ldm_ids": [1, 2, 3],
            "ldm_flip_order": [3, 2, 1],
            "ldm_edges_matrix": [[0, 1], [1, 2]],
            "num_landmarks": 3,
            "num_edges": 1,
            "fields": ["imgpath", "bbox", "headpose", "ids", "landmarks", "visible"],
        }
        self.assertEqual(state_dict, expected_state_dict)

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_str_representation(self, mock_file, mock_exists):
        mock_exists.return_value = True
        mock_data = {
            "ldm_ids": [1, 2, 3],
            "ldm_flip_order": [3, 2, 1],
            "ldm_edges_matrix": [[0, 1], [1, 2]],
        }
        mock_file.return_value.read.return_value = json.dumps(mock_data)

        db = DatabaseStruct("test_db")
        str_repr = str(db)
        self.assertIn("Database {", str_repr)
        self.assertIn("name: test_db", str_repr)
        self.assertIn("num_landmarks: 3", str_repr)
        self.assertIn("num_edges: 1", str_repr)
