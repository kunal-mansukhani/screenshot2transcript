import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import visualize_mask_distribution
import torchvision

class SpeechBubbleDataset(Dataset):
    def __init__(self, data_dir, mode, img_transform=None, mask_transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        if mode == "test":
            self.image_dir = os.path.join(data_dir, mode, 'dataset')
            self.mask_dir = os.path.join(data_dir, mode, 'masks')
        else:
            self.image_dir = os.path.join(data_dir, mode, 'augmented_dataset')
            self.mask_dir = os.path.join(data_dir, mode, 'augmented_masks')
        self.image_filenames = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        
        # Generate the corresponding mask filename
        mask_filename = os.path.splitext(image_filename)[0] + '_mask.png'
        
        mask_path = os.path.join(self.mask_dir, mask_filename)
        # Open and convert the image
        image = Image.open(image_path).convert('RGB')
        #print(image.size)
        # Open and convert the mask
        mask = Image.open(mask_path).convert('L')
        
        freq = [0, 0, 0]
        for pixel in mask.getdata():
            freq[pixel] += 1
            
        #print(f'Percent of background = {freq[0]/sum(freq)*100:.2f}%', f'Percent of her_bubble = {freq[1]/sum(freq)#*100:.2f}%', f'Percent of you_bubble = {freq[2]/sum(freq)*100:.2f}%')
        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        

     
        return image, mask