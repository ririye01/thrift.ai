import pandas as pd
import io
from typing import Any, List, Dict
from PIL import Image
from datasets import load_dataset, DatasetDict, Dataset


# Helper function to convert PIL Image to bytes
def convert_image_to_bytes(image: Image.Image) -> bytes:
    img_byte_arr: io.BytesIO = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_bytes: bytes = img_byte_arr.getvalue()
    return img_bytes


# Transform and store datasets
def process_and_store_datasets(subset: Dataset, name: str) -> None:
    records: List[Dict[str, Any]] = []
    for data in subset:
        record: Dict[str, Any] = {
            "image_id": data["image_id"],
            "width": data["width"],
            "height": data["height"],
            "bbox": data["objects"]["bbox"],
            "category": data["objects"]["category"],
            "area": data["objects"].get("area"),
            # convert image to bytes and store directly
            "image": convert_image_to_bytes(data["image"]),
        }
        records.append(record)

    # Create a DataFrame
    df: pd.DataFrame = pd.DataFrame(records)

    # Save the DataFrame to a Parquet file
    df.to_parquet(f"data/huggingface/{name}_dataset.parquet")


if __name__ == "__main__":
    # Load your datasets
    dataset: DatasetDict = load_dataset("detection-datasets/fashionpedia")
    train_dataset: Dataset = dataset["train"]
    val_dataset: Dataset = dataset["val"]

    # Process both train and validation datasets
    process_and_store_datasets(train_dataset, "train")
    process_and_store_datasets(val_dataset, "val")
