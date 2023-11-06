import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
import io
from typing import List, Dict, Tuple

# Ensure the output directory exists
import os
output_dir = "data/example_images"
os.makedirs(output_dir, exist_ok=True)


# Function to draw bounding boxes and display images
def draw_bounding_boxes(
    image_bytes: bytes,
    bboxes: List[Tuple[float, float, float, float]],
    categories: List[int],
    category_mapping: Dict[int, str],
) -> Image.Image:
    image: Image.Image = Image.open(io.BytesIO(image_bytes))
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)

    # Draw each bbox
    for bbox, category in zip(bboxes, categories):
        draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline="red", width=3)
        category_name: str = category_mapping.get(category, "Unknown")
        draw.text((bbox[0], bbox[1]), category_name, fill="blue")

    return image


if __name__ == "__main__":
    # Replace with your actual categories - this is just an example
    category_mapping: Dict[int, str] = {
        0: "shirt, blouse",
        1: "top, t-shirt, sweatshirt",
        2: "sweater",
        3: "cardigan",
        4: "jacket",
        5: "vest",
        6: "pants",
        7: "shorts",
        8: "skirt",
        9: "coat",
        10: "dress",
        11: "jumpsuit",
        12: "cape",
        13: "glasses",
        14: "hat",
        15: "headband, head covering, hair accessory",
        16: "tie",
        17: "glove",
        18: "watch",
        19: "belt",
        20: "leg warmer",
        21: "tights, stockings",
        22: "sock",
        23: "shoe",
        24: "bag, wallet",
        25: "scarf",
        26: "umbrella",
        27: "hood",
        28: "collar",
        29: "lapel",
        30: "epaulette",
        31: "sleeve",
        32: "pocket",
        33: "neckline",
        34: "buckle",
        35: "zipper",
        36: "applique",
        37: "bead",
        38: "bow",
        39: "flower",
        40: "fringe",
        41: "ribbon",
        42: "rivet",
        43: "ruffle",
        44: "sequin",
        45: "tassel",
    }

    # Open the Parquet file
    parquet_file = pq.ParquetFile("data/huggingface/train_dataset.parquet")

    # Initialize an empty DataFrame
    df = pd.DataFrame()

    # Batch size
    BATCH_SIZE: int = 25

    # Read the Parquet file in smaller batches
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["image", "bbox", "category"]):
        batch_df = batch.to_pandas()
        df = pd.concat([df, batch_df], ignore_index=True)
        if len(df) >= BATCH_SIZE:  # Only read first 10 records
            df = df.head(BATCH_SIZE)
            break


    # Iterate over the rows and process each image
    for index, row in df.iterrows():
        image_with_boxes: Image.Image = draw_bounding_boxes(
            row["image"],
            row["bbox"],
            row["category"],
            category_mapping,
        )
        image_path = os.path.join(output_dir, f"image_{index}.jpeg")
        image_with_boxes.save(image_path)
        print(f"Saved: {image_path}")
