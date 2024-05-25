import tensorflow as tf
import json
import os
from object_detection.utils import dataset_util

# Paths
training_annotations_path = 'data/training/labels.json'
validation_annotations_path = 'data/validation/labels.json'
training_images_dir = 'data/training/images'
validation_images_dir = 'data/validation/images'
training_tfrecord_path = 'data/training/training.tfrecord'
validation_tfrecord_path = 'data/validation/validation.tfrecord'

# Read annotations
def load_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        return json.load(f)

training_annotations = load_annotations(training_annotations_path)
validation_annotations = load_annotations(validation_annotations_path)

# Helper functions to create TFRecord
def create_tf_example(image, annotations, image_dir):
    img_path = os.path.join(image_dir, image['file_name'])
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_image_data = fid.read()
    
    width = int(image['width'])
    height = int(image['height'])
    
    filename = image['file_name'].encode('utf8')
    image_format = b'png' if image['file_name'].endswith('.png') else b'jpg'
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for ann in annotations:
        xmins.append(ann['bbox'][0] / width)
        ymins.append(ann['bbox'][1] / height)
        xmaxs.append((ann['bbox'][0] + ann['bbox'][2]) / width)
        ymaxs.append((ann['bbox'][1] + ann['bbox'][3]) / height)
        classes_text.append(str(ann['category_id']).encode('utf8'))
        classes.append(ann['category_id'])
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_tfrecord(annotations, image_dir, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        for image in annotations['images']:
            image_id = image['id']
            img_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
            tf_example = create_tf_example(image, img_annotations, image_dir)
            writer.write(tf_example.SerializeToString())

# Create TFRecords
create_tfrecord(training_annotations, training_images_dir, training_tfrecord_path)
create_tfrecord(validation_annotations, validation_images_dir, validation_tfrecord_path)

print('TFRecord files created successfully.')
