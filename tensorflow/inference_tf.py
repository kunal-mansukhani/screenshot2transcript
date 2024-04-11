import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    """Load an image file, resize it, convert it to numpy array, and preprocess it for model inference."""
    img = Image.open(image_path).resize((256, 256))
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize the image
    if img_array.shape[-1] == 4:  # Check if the image has an alpha channel
        img_array = img_array[..., :3]  # Remove alpha channel if present
    img_array = np.transpose(img_array, (2, 0, 1))  # Transpose from HWC to CHW format
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def run_inference(model_directory, image_path):
    """Run model inference using the SavedModel located in model_directory on the image specified by image_path."""
    model = tf.saved_model.load(model_directory)
    infer = model.signatures['serving_default']
    
    input_tensor = load_and_preprocess_image(image_path)
    output = infer(tf.constant(input_tensor))  # Run the model
    return output

# Use the function
model_directory = 'model_tf'
image_path = 'test2.jpeg'
output = run_inference(model_directory, image_path)

# Assuming the output tensor name is 'output_0', change according to your model's output
output_data = output['output'].numpy()

def visualize_output(output_data):
    """Assuming output_data is a 2D array, visualize it."""
    plt.imshow(output_data[0, 0, :, :], cmap='viridis')  # Visualize the first channel of the output
    plt.axis('off')
    plt.show()

visualize_output(output_data)
