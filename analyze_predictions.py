#file for analyzing the predictions

import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from data_loader import full_dataset, valid_loader, device

def visualize_gradcam_predictions(model, loader, num_samples=3):
    model.eval()
    
    # Initialize containers for different prediction categories
    samples = {
        'correct_0': [],    # Correct predictions for class 0
        'correct_1': [],    # Correct predictions for class 1
        'incorrect_0': [],  # Incorrect predictions for class 0
        'incorrect_1': []   # Incorrect predictions for class 1
    }
    
    # Setup GradCAM - removed use_cuda parameter
    target_layers = [model.features[-2]]  # Using the last convolutional layer
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Rest of the function remains the same
    with torch.no_grad():
        for images, labels in loader:
            if all(len(samples[key]) >= num_samples for key in samples):
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for idx, (image, label, pred) in enumerate(zip(images, labels, predicted)):
                category = f"{'correct' if label == pred else 'incorrect'}_{label.item()}"
                
                if len(samples[category]) < num_samples:
                    samples[category].append({
                        'image': image.cpu(),
                        'true': label.item(),
                        'pred': pred.item()
                    })
    
    # Visualize samples with GradCAM
    class_names = list(full_dataset.class_to_idx.keys())
    fig_rows = 4  # One row for each category
    fig = plt.figure(figsize=(15, 5 * fig_rows))
    
    for row, (category, category_samples) in enumerate(samples.items()):
        for col, sample in enumerate(category_samples):
            # Original image
            ax = plt.subplot(fig_rows, num_samples * 2, row * (num_samples * 2) + col * 2 + 1)
            img = sample['image'].numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            plt.imshow(img)
            plt.axis('off')
            if col == 0:
                plt.ylabel(category.replace('_', ' ').title())
            plt.title(f'True: {class_names[sample["true"]]}\nPred: {class_names[sample["pred"]]}')
            
            # GradCAM visualization
            ax = plt.subplot(fig_rows, num_samples * 2, row * (num_samples * 2) + col * 2 + 2)
            input_tensor = sample['image'].unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(sample['pred'])]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            plt.imshow(visualization)
            plt.axis('off')
            plt.title('GradCAM')
    
    plt.tight_layout()
    plt.show()

# Run the visualization

