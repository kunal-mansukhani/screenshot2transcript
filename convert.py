import numpy as np
import torch
from model import BubbleNet
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

def representative_dataset_gen():
    for _ in range(100):
        yield [np.random.rand(1, 256, 256, 3).astype(np.float32)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BubbleNet().to(device)
model.load_state_dict(torch.load('checkpoints/bubble_classifier_mobile.pth', map_location=device))
model.eval()

dummy_input = torch.randn(1, 3, 256, 256, device=device)
torch.onnx.export(model, dummy_input, 'model.onnx', export_params=True, opset_version=12, do_constant_folding=True, input_names=['input'], output_names=['output'])

onnx_model = onnx.load('model.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('model_tf')

converter = tf.lite.TFLiteConverter.from_saved_model('model_tf')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

try:
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite model has been saved successfully.")
except Exception as e:
    print(f"Failed to convert TF to TFLite due to: {e}")
