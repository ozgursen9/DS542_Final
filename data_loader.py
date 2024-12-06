#file for loading and preprocessing the data

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

# Define constants
IMG_SIZE = 256
BATCH_SIZE = 32
DATA_DIR = 'Dataset'
VALID_SPLIT = 0.2

# Define the transformations for training (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, translate=(0.2, 0.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Define the transformations for validation (no augmentation)
valid_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load the datasets
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)

# Calculate lengths for train/validation split
total_size = len(full_dataset)
valid_size = int(VALID_SPLIT * total_size)
train_size = total_size - valid_size

# Split the dataset
train_dataset, valid_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, valid_size]
)

# Override the transform for validation dataset
valid_dataset.dataset.transform = valid_transform

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

# Print some information about the datasets
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}")
print(f"Class labels: {full_dataset.class_to_idx}")

# Optional: check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def show_random_images(dataset, num_images=5, title=""):
    # Create a figure with num_images columns
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    fig.suptitle(title)
    
    # Get random indices
    indices = random.sample(range(len(dataset)), num_images)
    
    for idx, ax in zip(indices, axes):
        # Get image and label
        image, label = dataset[idx]
        
        # Convert tensor to numpy and transpose from (C,H,W) to (H,W,C)
        image = image.numpy().transpose(1, 2, 0)
        
        # Denormalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        
        # Clip values to be between 0 and 1
        image = np.clip(image, 0, 1)
        
        # Plot the image
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f'Label: {list(full_dataset.class_to_idx.keys())[label]}')

def show_random_images(dataset, num_images=5, title=""):
    # Create a figure with num_images columns
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    fig.suptitle(title)
    
    # Get random indices
    indices = random.sample(range(len(dataset)), num_images)
    
    for idx, ax in zip(indices, axes):
        # Get image and label
        image, label = dataset[idx]
        
        # Convert tensor to numpy and transpose from (C,H,W) to (H,W,C)
        image = image.numpy().transpose(1, 2, 0)
        
        # Denormalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        
        # Clip values to be between 0 and 1
        image = np.clip(image, 0, 1)
        
        # Plot the image
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f'Label: {list(full_dataset.class_to_idx.keys())[label]}')
