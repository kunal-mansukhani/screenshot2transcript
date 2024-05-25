import json
import os
import shutil
import random

# Paths
original_annotations_path = 'data/train/annotations/result.json'
original_images_dir = 'data/train/annotations/images'
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
split_index = int(len(images) * 0.85)
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

# Copy images to new directories
def copy_images(images, source_dir, target_dir):
    for image in images:
        src_path = os.path.join(source_dir, image['file_name'])
        dst_path = os.path.join(target_dir, image['file_name'])
        shutil.copy(src_path, dst_path)

copy_images(training_images, original_images_dir, training_images_dir)
copy_images(validation_images, original_images_dir, validation_images_dir)

print('Dataset split and files created successfully.')
