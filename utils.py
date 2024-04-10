import torch
import matplotlib.pyplot as plt
from PIL import Image
def visualize_mask_distribution(mask):
    mask = mask.cpu()
    # Flatten the mask tensor to make counting easier
    flat_mask = mask.view(-1)

    # Create a histogram of class labels
    hist = torch.histc(flat_mask.float(), bins=3, min=0, max=255)

    # Create a bar plot
    class_labels = ['Background', 'Her Bubble', 'You Bubble']
    plt.figure(figsize=(8, 6))
    plt.bar(class_labels, hist.numpy())
    plt.xlabel('Class')
    plt.ylabel('Pixel Count')
    plt.title('Mask Class Distribution')
    plt.show()
    
def calculate_class_percentages(mask):
    # Flatten the mask tensor to make counting easier
    flat_mask = mask.view(-1)

    # Calculate total number of pixels
    total_pixels = flat_mask.size(0)

    # Calculate counts of each class
    background_count = torch.sum(flat_mask == 0).item()
    her_bubble_count = torch.sum(flat_mask == 1).item()
    you_bubble_count = torch.sum(flat_mask == 2).item()
    print(f"total pixels: {total_pixels}")
    # Calculate percentages
    background_percentage = (background_count / total_pixels) * 100
    her_bubble_percentage = (her_bubble_count / total_pixels) * 100
    you_bubble_percentage = (you_bubble_count / total_pixels) * 100

    return background_percentage, her_bubble_percentage, you_bubble_percentage
def count_class_predictions(output_tensor):
    """
    Counts the number of times each class has the highest logit in the output tensor.

    Parameters:
    output_tensor (torch.Tensor): A tensor of size (batch_size, num_classes, height, width)
                                  containing raw logits.

    Returns:
    torch.Tensor: A tensor containing counts for each class.
    """
    # Get the indices of the max logit.
    # The result will have the same size as the height x width, containing the index of the max logit.
    class_predictions = torch.argmax(output_tensor, dim=1)
    
    # Count the occurrences of each class in the predictions.
    counts = torch.stack([(class_predictions == class_index).sum() for class_index in range(output_tensor.size(1))])

    return counts

def draw_segmentation_map(predictions, device):
    """
    Draw the segmentation map from the model's predictions.
    Parameters:
        predictions (torch.Tensor): The output tensor from the model.
                                    Shape: [1, num_classes, height, width].
        device: The device to perform computations on.
    Returns:
        PIL.Image: The segmentation map as a PIL Image.
    """
    predictions = predictions.squeeze(0)  # Remove batch dim: [num_classes, H, W]
    
    # Convert the predictions to class indices [H, W]
    class_indices = torch.argmax(predictions, dim=0)
    
    # Create a colors tensor and move it to the same device as predictions
    colors = torch.tensor([[0, 0, 0], [255, 105, 180], [0, 0, 255]], dtype=torch.uint8).to(device)
    
    # Initialize an empty tensor for the colored mask
    colored_mask = torch.zeros((3, *class_indices.shape), dtype=torch.uint8).to(device)
    
    # Map each class index to a color
    for i in range(colors.size(0)):  # Iterate through each color (class)
        mask = class_indices == i
        colored_mask[:, mask] = colors[i][:, None]

    # Convert the colored mask to a PIL Image
    # Since operations with PIL Images assume CPU, move the tensor back to CPU
    colored_mask_image = Image.fromarray(colored_mask.permute(1, 2, 0).cpu().numpy(), mode='RGB')

    return colored_mask_image