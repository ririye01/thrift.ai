from typing import List, Dict, Any, Union

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
        data = response.json()
        self._annotations = data["annotations"]


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
            self._load_annotations()
        return self._annotations