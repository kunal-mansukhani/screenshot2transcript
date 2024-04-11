import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

def softmax(x):
    """Compute softmax values for each sets of scores in x over the last dimension."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def load_image_into_numpy_array(path):
    """Loads an image and converts it to a numpy array."""
    img = Image.open(path).resize((256, 256))
    img_array = np.array(img, dtype=np.float32) / 255.0
    if img_array.shape[-1] == 4:  # Drop the alpha channel if present
        img_array = img_array[..., :3]
    return np.transpose(img_array, (2, 0, 1))  # CHW format for TFLite compatibility

def run_tflite_model(tflite_file, test_image_path):
    """Runs the TensorFlow Lite model on the given test image."""
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare the input image
    input_data = load_image_into_numpy_array(test_image_path)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension (NCHW)
    
    # Set the model input and invoke the interpreter
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get the model output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data.squeeze(0)  # Remove the batch dimension, retain CHW

def visualize_segmentation(segmentation_map, title="Segmentation Map", save_path='segmentation_output.png'):
    """Visualizes the segmentation map."""
    plt.figure(figsize=(10, 5))
    plt.imshow(segmentation_map, interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path)
    plt.show()

# File paths
tflite_model_path = 'model.tflite'
test_image_path = 'test_1.jpg'

# Run the model and process the output
output = run_tflite_model(tflite_model_path, test_image_path)
print("Raw model output shape:", output.shape)

# Convert logits to probabilities and find the predicted class per pixel
probabilities = softmax(output)
print(probabilities)
predicted_classes = np.argmax(probabilities, axis=0)  # Use axis=0 assuming output shape is CHW

# Visualize the segmentation map
visualize_segmentation(predicted_classes)
