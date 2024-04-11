import torch
import torchvision.transforms as transforms
from PIL import Image
from model import BubbleNet  
from utils import count_class_predictions
import numpy as np
from PIL import Image
from utils import draw_segmentation_map
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BubbleNet().to(device)
model.load_state_dict(torch.load('checkpoints/best_bubble_classifier.pth', map_location=device))  
model.eval()

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

image_path = 'test2.jpeg' 
image = Image.open(image_path).convert('RGB')
image = image_transform(image)
image = image.unsqueeze(0)  # image is now of shape [1, 3, 512, 512]


image = image.to(device)

with torch.no_grad():  # No need to track gradients for inference
    output = model(image)
    # The output here is the raw logits. You need to apply softmax to get probabilities
    print(output.shape)
    preds = count_class_predictions(output)
    print(preds)
    probabilities = torch.softmax(output, dim=1)
    segmentation_map = draw_segmentation_map(probabilities, device)
    segmentation_map.show()
    segmentation_map.save('segmentation_output.png')
    # Move the image to the device

