# Imports 
from pathlib import Path
from negativeimages import ANC_PATH, POS_PATH, NEG_PATH
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
from torchvision import transforms



# Define a custom dataset 
class SiameseDataset(Dataset):
    def __init__(self, anchor_path, comparison_path, labels):
        self.anchor = anchor_path
        self.comparison = comparison_path
        self.labels = labels 

    def __len__(self):
        return len(self.anchor)
    
    def __getitem__(self, idx):
        return preprocess(self.anchor[idx]), preprocess(self.comparison[idx]), self.labels[idx]
    

# Load images from a dir 
def load_image_paths(directory: str, limit: int = 300) -> list[str]:
    """
    Load image file paths from a directory with a limit
  
    Returns:
        List of image file paths
    """
    # Convert directory to Path object and get all jpg files
    path = Path(directory)
    all_images = list(path.glob('*.jpg'))
    
    # Take only the first 'limit' images (randomly shuffle if you want random selection)
    selected_images = all_images[:limit] if len(all_images) > limit else all_images
    
    # Convert Path objects to strings
    return [str(img_path) for img_path in selected_images]

# Define a preprocess function 
def preprocess(file_path):
    """
    Load and preprocess an image from a file path

    Returns:
        Preprocessed image as a torch tensor
    """
    # Read and load the image using PIL
    img = Image.open(file_path)
    
    # Define preprocessing steps
    transform = transforms.Compose([
        # Resize to 100x100
        transforms.Resize((105, 105)),  
        transforms.ToTensor()
    ])
    
    # Apply transformations
    return transform(img)


# Load data
limit = 600
anchor = load_image_paths(ANC_PATH, limit=limit) # Load #limit anchor images
positive = load_image_paths(POS_PATH, limit=limit) # Load #limit positive image
negative = load_image_paths(NEG_PATH, limit=limit) # Load #limit negative images

# Make the positives Dataset
positives = SiameseDataset( 
anchor_path = anchor,
comparison_path = positive,
labels = torch.ones(len(anchor))
)

# Make the negatives Dataset
negatives = SiameseDataset(
    anchor_path = anchor,
    comparison_path = negative,
    labels = torch.zeros(len(anchor))
)

# Combine both the positives and negatives
data = ConcatDataset([positives, negatives])

# Train test split - 80/20
train_length = int(0.8 * len(data))
test_length = int(len(data) - train_length)

# Perform split
train_data, test_data = random_split(data, [train_length, test_length])

# Batch Size
batch_size = 18

# Create the train and test dataloaders 
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle = False)
