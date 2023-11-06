# Standard library imports
import json
from typing import List, Dict, Any, Union

# Related third-party imports
import numpy as np
import requests


class AnnotationsLoader:
    def __init__(
        self, 
        training_annotations_route: str = "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json",
        validation_annotations_route: str = "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json",
        testing_annotations_route: str = "https://s3.amazonaws.com/ifashionist-dataset/annotations/info_test2020.json",
        training_attribute_ids_route: str = "https://s3.amazonaws.com/ifashionist-dataset/annotations/attributes_train2020.json",
        validation_attribute_ids_route: str = "https://s3.amazonaws.com/ifashionist-dataset/annotations/attributes_val2020.json",
    ) -> None:
        # Training data attributes
        self._training_annotations_route: str = training_annotations_route
        self._training_attribute_ids_route: str = training_attribute_ids_route
        self._training_annotations: Union[List[Dict[str, Any]], None] = None
        self._training_formatted_annotations: Union[np.ndarray, None] = None

        # Validation data attributes
        self._validation_annotations_route: str = validation_annotations_route
        self._validation_attribute_ids_route: str = validation_attribute_ids_route
        self._validation_annotations: Union[List[Dict[str, Any]], None] = None
        self._validation_formatted_annotations: Union[np.ndarray, None] = None

        # Testing data attributes
        self._testing_annotations_route: str = testing_annotations_route
        self._testing_annotations: Union[List[Dict[str, Any]], None] = None
        self._testing_formatted_annotations: Union[np.ndarray, None] = None


    def _load_training_annotations(self) -> None:
        """
        Load all of the annotations for the training images from the JSON file.
        This method is intended to be private and should only be called internally.
        """
        response = requests.get(self._training_annotations_route)
        response.raise_for_status()
        self._training_annotations = response.json()


    @property
    def training_annotations(self) -> List[Dict[str, Any]]:
        """
        A getter for the annotations data. It lazily loads the data if not already loaded.

        Returns
        -------
        List[Dict[str, Any]]
            The loaded annotations data.
        """
        if self._training_annotations is None:
            self._load_training_annotations()
        return self._training_annotations


    def _load_json(self, route: str) -> Dict[str, Any]:
        """
        Load JSON data from the specified route.
        """
        response = requests.get(route)
        response.raise_for_status()
        return response.json()


    def get_json_keys(self, route: str) -> List[str]:
        """
        Get the top-level keys of the JSON response from the specified route.
        """
        json_data = self._load_json(route)
        return list(json_data.keys())


    def get_json_schema(self, route: str, depth: int = 10) -> str:
        """
        Get the schema of the JSON response, showing nested keys up to the specified depth,
        and return it as a pretty-printed string.
        """
        def get_schema_recursive(json_obj: Union[Dict, List], current_depth: int) -> Union[Dict[str, Any], str]:
            if isinstance(json_obj, dict):
                return {k: get_schema_recursive(v, current_depth - 1) for k, v in json_obj.items()} if current_depth > 0 else '...'
            elif isinstance(json_obj, list):
                return [get_schema_recursive(json_obj[0], current_depth - 1)] if json_obj else []
            else:
                return type(json_obj).__name__

        json_data = self._load_json(route)
        schema = get_schema_recursive(json_data, depth)
        return json.dumps(schema, indent=2)  # Use `json.dumps` to pretty print the schema