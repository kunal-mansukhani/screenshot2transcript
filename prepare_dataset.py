import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import os
import shutil

class_labels = {'background': 0, 'her_bubble': 1, 'you_bubble': 2}

def generate_masks(annotations_file, mode):
    tree = ET.parse(annotations_file)
    root = tree.getroot()

    masks_dir = f'data/{mode}/masks'
    colored_masks_dir = f'data/{mode}/masks_colored'

    if os.path.exists(masks_dir):
        shutil.rmtree(masks_dir)
    if os.path.exists(colored_masks_dir):
        shutil.rmtree(colored_masks_dir)

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(colored_masks_dir, exist_ok=True)

    for image in root.iter('image'):
        if image.attrib['subset'].lower() == mode:
            image_name = os.path.splitext(image.attrib['name'])[0]
            width = int(image.attrib['width'])
            height = int(image.attrib['height'])

            mask = np.zeros((height, width), dtype=np.uint8)

            for box in image.iter('box'):
                label = class_labels[box.attrib['label']]

                xtl = int(float(box.attrib['xtl']))
                ytl = int(float(box.attrib['ytl']))
                xbr = int(float(box.attrib['xbr']))
                ybr = int(float(box.attrib['ybr']))

                mask[ytl:ybr, xtl:xbr] = label

            mask_img = Image.fromarray(mask)

            mask_filename = f'{image_name}_mask.png'
            mask_img.save(os.path.join(masks_dir, mask_filename))

            colored_mask_img = Image.fromarray(mask * 40)  
            colored_mask_filename = f'{image_name}_mask_colored.png'
            colored_mask_img.save(os.path.join(colored_masks_dir, colored_mask_filename))

generate_masks('data/annotations.xml', 'train')
generate_masks('data/annotations.xml', 'test')