import torch
from model import BubbleNet
from torch.quantization import quantize_dynamic
import torch.utils.mobile_optimizer as mobile_optimizer

model = BubbleNet(num_classes=3, pretrained=True)  # Choose the appropriate model version
model.load_state_dict(torch.load('checkpoints/bubble_classifier_mobile.pth'))
model.eval()

# Convert the model to a ScriptModule
scripted_model = torch.jit.script(model)

# Proceed with dynamic quantization on the scripted model
quantized_model = quantize_dynamic(scripted_model, dtype=torch.qint8)

# Optimize the quantized model for mobile
mobile_optimizer.optimize_for_mobile(quantized_model)

# Save the quantized model state dictionary
torch.save(quantized_model.state_dict(), 'checkpoints/quantized_model.pth')

# Trace the quantized model
traced_model = torch.jit.trace(quantized_model, torch.zeros((1, 3, 256, 256)))

# Save the traced model
traced_model.save('checkpoints/quantized_model.pt')