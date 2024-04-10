import os
from PIL import Image, ImageOps

# Define the paths to the image and mask directories
image_dir = 'data/train/dataset'
mask_dir = 'data/train/masks'

# Create directories for the augmented images and masks
aug_image_dir = 'data/train/augmented_dataset'
aug_mask_dir = 'data/train/augmented_masks'
os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_mask_dir, exist_ok=True)

# Iterate over each image in the image directory


for image_file in os.listdir(image_dir):
    if image_file == ".DS_Store":
        continue
    # Get the image name without the extension
    image_name = os.path.splitext(image_file)[0]
    
    # Load the image and corresponding mask
    image_path = os.path.join(image_dir, image_file)
    mask_path = os.path.join(mask_dir, f"{image_name}_mask.png")
    image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB mode
    mask = Image.open(mask_path)
    
    # Save the original image and mask to the augmented directories
    image.save(os.path.join(aug_image_dir, f"{image_name}.png"))
    mask.save(os.path.join(aug_mask_dir, f"{image_name}_mask.png"))
    
    # Create a grayscaled version of the image
    grayscale_image = ImageOps.grayscale(image)
    grayscale_image.save(os.path.join(aug_image_dir, f"{image_name}_grayscale.png"))
    mask.save(os.path.join(aug_mask_dir, f"{image_name}_grayscale_mask.png"))
    
    # Create an inverted color version of the image
    inverted_image = ImageOps.invert(image)  # Now works since image is guaranteed to be in "RGB"
    inverted_image.save(os.path.join(aug_image_dir, f"{image_name}_inverted.png"))
    mask.save(os.path.join(aug_mask_dir, f"{image_name}_inverted_mask.png"))
