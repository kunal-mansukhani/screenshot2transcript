import torch
from model import BubbleNet  # Ensure this is your PyTorch model. It should be correctly defined in the model.py file.

# Instantiate your model
model = BubbleNet()
model.load_state_dict(torch.load('checkpoints/bubble_classifier_mobile.pth', map_location=torch.device('cpu')))  # Ensure the model and checkpoint are compatible
model.eval()

# Create dummy input that matches the model's input shape
dummy_input = torch.randn(1, 3, 256, 256)  # Adjust the dimensions as per your model's requirements

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, 'model.onnx', export_params=True, opset_version=12, do_constant_folding=True)
import onnx
from onnx_tf.backend import prepare

# Load the ONNX model
onnx_model = onnx.load('model.onnx')

# Prepare the TensorFlow representation
tf_rep = prepare(onnx_model)

# Export the model as a TensorFlow graph
tf_rep.export_graph('model_tf')
import tensorflow as tf

# Convert the TensorFlow model directory to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_saved_model('model_tf')
tflite_model = converter.convert()

# Save the TFLite model to disk
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
