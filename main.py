# Standard library imports
from typing import Tuple, Any

# Related third-party imports
import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset

# Local application/library specific imports
from data_loading.annotation_loader import AnnotationsLoader


def load_fashionpedia_training_data() -> Tuple[Any, Any]:
    """
    Load all necessary data for
    """
    annotations_loader = AnnotationsLoader()
    tmp = annotations_loader.load_training_data_annotations()
    print(tmp[:10])
    return ("images", "data_representation")


def get_fashionpedia_json_schemas():
    annotations_loader = AnnotationsLoader()

    # Get top-level keys of the training annotations JSON
    training_json_keys = annotations_loader.get_json_keys(
        annotations_loader._training_annotations_route
    )
    print("Training JSON keys:", training_json_keys)

    for key in [
        "info",
        "annotations",
        "images",
        "licenses",
        "categories",
        "attributes",
    ]:
        if key == "info":
            print(annotations_loader.training_annotations[key])

    # Get schema of the training annotations JSON up to a depth of 5
    training_json_schema = annotations_loader.get_json_schema(
        annotations_loader._training_annotations_route, depth=5
    )
    print("Training JSON schema:", training_json_schema, "\n\n")

    # Get top-level keys of the training annotations JSON
    training_id_json_keys = annotations_loader.get_json_keys(
        annotations_loader._training_attribute_ids_route
    )
    print("Training JSON keys:", training_id_json_keys)

    # Get schema of the training annotations JSON up to a depth of 2
    training_id_json_schema = annotations_loader.get_json_schema(
        annotations_loader._training_attribute_ids_route, depth=5
    )
    print("Training JSON schema:", training_id_json_schema)


if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("detection-datasets/fashionpedia")

    # The dataset object is a dictionary with all the splits available
    # For example, to access the train split you can do
    train_dataset = dataset["train"]

    # To iterate over the dataset
    for sample in train_dataset:
        print(sample)
        # Do something with the sample

    # To access a specific element in the dataset
    print(train_dataset[0])
