import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def parse_tfrecord_fn(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example['image/encoded'], channels=3)
    xmin = tf.sparse.to_dense(example['image/object/bbox/xmin'])
    xmax = tf.sparse.to_dense(example['image/object/bbox/xmax'])
    ymin = tf.sparse.to_dense(example['image/object/bbox/ymin'])
    ymax = tf.sparse.to_dense(example['image/object/bbox/ymax'])
    labels = tf.sparse.to_dense(example['image/object/class/label'])
    
    # Ensure bounding box coordinates are normalized between 0 and 1
    xmin = tf.clip_by_value(xmin, 0.0, 1.0)
    xmax = tf.clip_by_value(xmax, 0.0, 1.0)
    ymin = tf.clip_by_value(ymin, 0.0, 1.0)
    ymax = tf.clip_by_value(ymax, 0.0, 1.0)

    bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=1)
    return image, bboxes, labels

def visualize_tfrecords(tfrecord_path, num_samples=5):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

    for image, bboxes, labels in parsed_dataset.take(num_samples):
        image = image.numpy()
        bboxes = bboxes.numpy()
        labels = labels.numpy()

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()

        im_height, im_width, _ = image.shape
        for bbox, label in zip(bboxes, labels):
            xmin, ymin, xmax, ymax = bbox
            xmin = int(xmin * im_width)
            xmax = int(xmax * im_width)
            ymin = int(ymin * im_height)
            ymax = int(ymax * im_height)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            plt.text(xmin, ymin, str(label), color='yellow', fontsize=12)

        plt.show()

# Path to your TFRecord file
tfrecord_path = '/content/data/training/training.tfrecord'
visualize_tfrecords(tfrecord_path, num_samples=5)
