import json
import os
import csv
import random

# Path to your COCO annotations JSON file
coco_annotations_file = 'data/train/annotations/result.json'
# Path to the directory containing your images
images_dir = 'images'
# Path to the output CSV file
output_csv_file = 'data/train/annotations/output.csv'

# Load the COCO annotations
with open(coco_annotations_file, 'r') as f:
    coco_data = json.load(f)

# Create a mapping from image ID to image file name
image_id_to_file_name = {image['id']: image['file_name'] for image in coco_data['images']}

# Shuffle annotations for random split
random.shuffle(coco_data['annotations'])

# Calculate split index
split_index = int(len(coco_data['annotations']) * 0.8)

# Open the output CSV file
with open(output_csv_file, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header
    csv_writer.writerow(['SET', 'IMAGE', 'LABEL', 'XMIN', 'YMIN', 'XMAX', 'YMAX'])

    # Write the annotations
    for i, annotation in enumerate(coco_data['annotations']):
        image_id = annotation['image_id']
        file_name = image_id_to_file_name[image_id]
        image_path = os.path.join(images_dir, file_name)

        label = annotation['category_id']
        xmin, ymin, width, height = annotation['bbox']
        xmax = xmin + width
        ymax = ymin + height

        # Determine if this annotation goes to training or validation set
        set_type = 'TRAIN' if i < split_index else 'VALIDATION'

        csv_writer.writerow([set_type, image_path, label, xmin, ymin, xmax, ymax])

print(f"CSV file has been created at {output_csv_file}")
