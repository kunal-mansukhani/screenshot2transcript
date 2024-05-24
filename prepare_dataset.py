import json
import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import shutil
import cv2

# Define your class labels and corresponding colors for visualization
class_labels = {'her_bubble': 0, 'you_bubble': 1, 'background': 2}
class_colors = {0: 128, 1: 255, 2: 0}  # Example grayscale colors for classes

def create_masks(coco_json_path, images_dir, masks_dir, colored_masks_dir, mode):
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    coco = COCO(coco_json_path)

    # Clean existing directories
    if os.path.exists(masks_dir):
        shutil.rmtree(masks_dir)
    if os.path.exists(colored_masks_dir):
        shutil.rmtree(colored_masks_dir)

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(colored_masks_dir, exist_ok=True)

    # Generate masks for each image
    for image_info in coco_data['images']:
        image_id = image_info['id']
        image_file = os.path.basename(image_info['file_name'])  # Get the image file name without the path
        image_path = os.path.join(images_dir, image_file)

        if not os.path.exists(image_path):
            print(f"Image {image_file} not found in {images_dir}. Skipping.")
            continue

        # Load the image to get dimensions
        image = Image.open(image_path)
        width, height = image.size

        # Initialize an empty mask with background class
        mask = np.full((height, width), class_labels['background'], dtype=np.uint8)
        print(f"Initialized mask for image {image_id} with background class.")

        # Get annotations for the current image
        annotation_ids = coco.getAnnIds(imgIds=[image_id])
        annotations = coco.loadAnns(annotation_ids)

        if not annotations:
            print(f"No annotations found for image {image_file}. Skipping.")
            continue

        for annotation in annotations:
            category_id = annotation['category_id']
            bbox = annotation['bbox']
            category_label = class_labels['her_bubble'] if category_id == 0 else class_labels['you_bubble']
            print(f"Processing annotation for image {image_id}, category {category_id}, label {category_label}. Bbox: {bbox}")

            x, y, w, h = map(int, bbox)
            cv2.rectangle(mask, (x, y), (x+w, y+h), color=category_label, thickness=-1)
            print(f"Filled bbox for category {category_label} in image {image_id}.")

        # Save the mask
        mask_img = Image.fromarray(mask)
        mask_filename = os.path.splitext(image_file)[0] + '_mask.png'
        mask_img.save(os.path.join(masks_dir, mask_filename))
        print(f'Saved mask for image {image_id} to {mask_filename}')

        # Debugging: Check unique values in the mask
        unique_values = np.unique(mask)
        print(f"Unique values in mask for image {image_id}: {unique_values}")

        # Save the colored mask for visualization
        colored_mask = np.zeros_like(mask, dtype=np.uint8)
        for cls_id, color in class_colors.items():
            colored_mask[mask == cls_id] = color
        print(f"Generated colored mask for image {image_id}.")

        # Debugging: Check unique values in the colored mask
        colored_unique_values = np.unique(colored_mask)
        print(f"Unique values in colored mask for image {image_id}: {colored_unique_values}")

        colored_mask_img = Image.fromarray(colored_mask)
        colored_mask_filename = os.path.splitext(image_file)[0] + '_mask_colored.png'
        colored_mask_img.save(os.path.join(colored_masks_dir, colored_mask_filename))
        print(f'Saved colored mask for image {image_id} to {colored_mask_filename}')

import os
from PIL import Image

# Define the directory containing the images

# List of supported extensions for conversion


def convert_to_png(image_dir):
    supported_extensions = ['.jpg', '.jpeg', '.webp']
    # Check if the image directory exists
    if not os.path.exists(image_dir):
        print(f"Directory '{image_dir}' does not exist.")
        return
    
    # Iterate over all files in the directory
    for filename in os.listdir(image_dir):
        # Get the file extension
        file_ext = os.path.splitext(filename)[1].lower()
        # Check if the file is a supported image type
        if file_ext in supported_extensions:
            # Define the full path to the image
            file_path = os.path.join(image_dir, filename)
            # Define the new filename with .png extension
            new_filename = os.path.splitext(filename)[0] + '.png'
            new_file_path = os.path.join(image_dir, new_filename)
            
            try:
                # Open the image file
                with Image.open(file_path) as img:
                    # Convert and save the image as PNG
                    img.save(new_file_path, 'PNG')
                    print(f"Converted {filename} to {new_filename}")
                    
                    # Verify the file is saved
                    if os.path.exists(new_file_path):
                        print(f"Successfully saved {new_filename}")
                        # Remove the original file
                        os.remove(file_path)
                        print(f"Removed original file {filename}")
                    else:
                        print(f"Failed to save {new_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")
if __name__ == "__main__":
    coco_json_path = 'data/train/annotations/result.json'  # Path to your COCO JSON file
    images_dir = 'data/train/annotations/images'                       # Path to your images directory
    masks_dir = 'data/train/masks'                         # Path to the directory where masks will be saved
    colored_masks_dir = 'data/train/masks_colored'         # Path to the directory where colored masks will be saved

    create_masks(coco_json_path, images_dir, masks_dir, colored_masks_dir, 'train')
    convert_to_png(images_dir)