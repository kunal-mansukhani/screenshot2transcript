import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SpeechBubbleDataset  
from model import BubbleNet
from utils import calculate_class_percentages, visualize_mask_distribution, count_class_predictions
from torchvision.transforms import Lambda
from PIL import Image
import numpy as np
from loss import DiceLoss
import matplotlib.pyplot as plt 

# Set device
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 50
batch_size = 4
learning_rate = 0.001
patience = 10

# Data preprocessing
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Default interpolation is bilinear, which is fine for images
    transforms.ToTensor(),  # Scales image data to [0, 1]
])

# Transform for masks
mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),  # Use nearest neighbor interpolation
    Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.int64))),  # Directly convert to Tensor without scaling
])
# Load the dataset
train_dataset = SpeechBubbleDataset('data', 'train', img_transform=image_transform, mask_transform=mask_transform)
val_dataset = SpeechBubbleDataset('data', 'test', img_transform=image_transform, mask_transform=mask_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Create an instance of the BubbleClassifier model
model = BubbleNet().to(device)

# Loss function and optimizer
cross_entropy_loss = nn.CrossEntropyLoss()
dice_loss = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

training_losses = []
validation_losses = []

best_val_loss = float('inf')
no_improve_epochs = 0
early_stopping_triggered = False


# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    step = 0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        #print(calculate_class_percentages(masks))
        optimizer.zero_grad()
        
        outputs = model(images)
        #print(count_class_predictions(outputs)) 
        loss = cross_entropy_loss(outputs, masks) + dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        step += 1
        
        if step % 4 == 0:
            print(f'Loss: {loss.item()}', f'Epoch: {epoch}', f'Step: {step}')
            
    
    epoch_train_loss = train_loss / len(train_dataset)
    
    training_losses.append(epoch_train_loss)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            
            outputs = model(images)
            loss =  cross_entropy_loss(outputs, masks) + dice_loss(outputs, masks)
            
            val_loss += loss.item() * images.size(0)
    
    epoch_val_loss = val_loss / len(val_dataset)
    validation_losses.append(epoch_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, LR: {current_lr}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), 'best_bubble_classifier.pth')
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping triggered")
            early_stopping_triggered = True
            break

if not early_stopping_triggered:
    print("Completed all epochs without early stopping.")

plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the trained model
print(f"Best validation loss: {best_val_loss}")
torch.save(model.state_dict(), 'checkpoints/bubble_classifier_mobile.pth')
