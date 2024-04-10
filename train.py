import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from model import BubbleNet  # Assuming BubbleNet is correctly defined for TensorFlow
from loss import DiceLoss  # Assuming dice_loss is correctly defined as per the DiceLoss class provided earlier

def load_image_and_mask(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = img / 255.0  # Normalize images to [0, 1]
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [256, 256], method='nearest')
    mask = tf.squeeze(mask)  # Remove channel dimension for SparseCategoricalCrossentropy
    
    return img, mask

def prepare_dataset(path, mask_path, batch_size):
    # List dataset files, accommodating various image formats
    image_dir = tf.io.gfile.glob(path + '/*.jpg') + tf.io.gfile.glob(path + '/*.jpeg') + \
                tf.io.gfile.glob(path + '/*.png') + tf.io.gfile.glob(path + '/*.webp')
    print(image_dir)

    # Generate mask filenames based on the image filenames
    mask_dir = [mask_path + '/' + tf.strings.split(tf.strings.split(s, '/')[-1], '.')[0] + '_mask.png' for s in image_dir]
    print(mask_dir)

    dataset = tf.data.Dataset.from_tensor_slices((image_dir, mask_dir))
    dataset = dataset.map(load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Paths to your data
train_image_path = 'data/train/augmented_dataset'  # Adjust these paths
train_mask_path = 'data/train/augmented_masks'  # Adjust these paths
test_image_path = 'data/test/dataset'  # Adjust these paths
test_mask_path = 'data/test/masks'  # Adjust these paths
train_dataset = prepare_dataset(train_image_path, train_mask_path, batch_size=4)
val_dataset = prepare_dataset(test_image_path, test_mask_path, batch_size=4)

dice_loss = DiceLoss()

model = BubbleNet(num_classes=3, input_shape=(256, 256, 3))
model.compile(
    optimizer=Adam(learning_rate=0.001, weight_decay=1e-5),
    loss=lambda y_true, y_pred: SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred) + dice_loss(y_true, y_pred),
    metrics=['accuracy']
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=50)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('checkpoints/bubble_classifier_mobile.h5')
