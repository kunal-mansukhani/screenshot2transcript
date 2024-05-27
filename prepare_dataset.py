import json
import os
import shutil
import random
from PIL import Image

# Paths
original_annotations_path = 'data/all/labels.json'
original_images_dir = 'data/all/images'
training_images_dir = 'data/training/images'
validation_images_dir = 'data/validation/images'
training_annotations_path = 'data/training/labels.json'
validation_annotations_path = 'data/validation/labels.json'

# Create directories
os.makedirs(training_images_dir, exist_ok=True)
os.makedirs(validation_images_dir, exist_ok=True)

# Read original annotations
with open(original_annotations_path, 'r') as f:
    annotations = json.load(f)

# Split images
images = annotations['images']
random.shuffle(images)
split_index = int(len(images) * 0.95)
training_images = images[:split_index]
validation_images = images[split_index:]

# Create image_id to annotation mapping
image_id_to_annotations = {}
for annotation in annotations['annotations']:
    image_id = annotation['image_id']
    if image_id not in image_id_to_annotations:
        image_id_to_annotations[image_id] = []
    image_id_to_annotations[image_id].append(annotation)

# Create new annotations
def create_annotations(images, image_id_to_annotations):
    new_annotations = {
        'images': images,
        'annotations': [],
        'categories': annotations['categories']
    }
    for image in images:
        image_id = image['id']
        if image_id in image_id_to_annotations:
            new_annotations['annotations'].extend(image_id_to_annotations[image_id])
    return new_annotations

training_annotations = create_annotations(training_images, image_id_to_annotations)
validation_annotations = create_annotations(validation_images, image_id_to_annotations)

# Write new annotations to files
with open(training_annotations_path, 'w') as f:
    json.dump(training_annotations, f)

with open(validation_annotations_path, 'w') as f:
    json.dump(validation_annotations, f)

# Convert images to .jpg and copy to new directories
def convert_and_copy_images(images, source_dir, target_dir):
    for image in images:
        src_path = os.path.join(source_dir, image['file_name'])
        file_name_without_ext, ext = os.path.splitext(image['file_name'])
        new_file_name = file_name_without_ext + '.jpg'
        dst_path = os.path.join(target_dir, new_file_name)
        
        # Convert image to .jpg
        with Image.open(src_path) as img:
            rgb_img = img.convert('RGB')
            rgb_img.save(dst_path, 'JPEG')

        # Update the file name in the annotations
        image['file_name'] = new_file_name

# Process and copy images
convert_and_copy_images(training_images, original_images_dir, training_images_dir)
convert_and_copy_images(validation_images, original_images_dir, validation_images_dir)

# Update the annotations with the new file names
with open(training_annotations_path, 'w') as f:
    json.dump(training_annotations, f)

with open(validation_annotations_path, 'w') as f:
    json.dump(validation_annotations, f)

print('Dataset split, files created, and images converted successfully.')
