# Standard library imports
from typing import Tuple, Any

# Related third-party imports
import cv2
import numpy as np
from PIL import Image

# Local application/library specific imports
from data_loading.annotation_loader import AnnotationsLoader


def load_fashionpedia_training_data() -> Tuple[Any, Any]:
    """
    Load all necessary data for 
    """
    tmp = load_training_data_annotations()
    print(tmp[:20])
    return ("images", "data_representation")


if __name__ == "__main__":
    load_fashionpedia_training_data()
