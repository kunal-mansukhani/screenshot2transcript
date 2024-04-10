import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold
from dataset import SpeechBubbleDataset
from model import BubbleNet

test_image_dir = 'test/dataset'
test_mask_dir = 'test/test_masks'

image_transform = transforms.Compose([
    transforms.Resize((512, 512)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

test_dataset = SpeechBubbleDataset(
    image_dir=test_image_dir,
    mask_dir=test_mask_dir,
    image_transform=image_transform,
    mask_transform=mask_transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

num_folds = 5
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda')  # Use 'cuda' if you have a GPU available

for fold in range(1, num_folds + 1):
    model = BubbleNet()
    model.load_state_dict(torch.load('speech_bubble_model.pth', map_location=device))
    model.to(device)
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            masks = masks.squeeze(1)  
            masks = masks.long()  
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            masks_upsampled = F.interpolate(masks.unsqueeze(1).float(), size=(1024, 1024), mode='nearest').long().squeeze(1)
            loss = criterion(outputs, masks_upsampled)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f'Fold [{fold}/{num_folds}], Test Loss: {test_loss:.4f}')