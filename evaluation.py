from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader import device, full_dataset, valid_loader

def evaluate_model(model, loader, num_misclassified=10):
    model.eval()
    all_preds = []
    all_labels = []
    misclassified_images = {0: [], 1: []}  # Store misclassified images for each class
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Store misclassified images
            misclassified_mask = (predicted != labels)
            misclassified_indices = misclassified_mask.nonzero()
            
            # Check if there are any misclassified images in this batch
            if misclassified_indices.numel() > 0:
                misclassified_indices = misclassified_indices.squeeze()
                # Handle case where only one misclassification is found
                if misclassified_indices.dim() == 0:
                    misclassified_indices = misclassified_indices.unsqueeze(0)
                
                for idx in misclassified_indices:
                    true_label = labels[idx].item()
                    if len(misclassified_images[true_label]) < num_misclassified:
                        misclassified_images[true_label].append({
                            'image': images[idx].cpu(),
                            'true': true_label,
                            'pred': predicted[idx].item()
                        })
    
    # Create and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    class_names = list(full_dataset.class_to_idx.keys())
    plt.xticks(ticks=[0.5, 1.5], labels=class_names)
    plt.yticks(ticks=[0.5, 1.5], labels=class_names)
    plt.show()
    
    # Print classification metrics
    total_samples = sum(sum(cm))
    accuracy = sum(cm[i][i] for i in range(len(cm))) / total_samples
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    # Calculate per-class metrics
    for i, class_name in enumerate(class_names):
        class_accuracy = cm[i][i] / sum(cm[i])
        print(f"{class_name} Accuracy: {class_accuracy:.2%}")
    
    # Plot misclassified images
    for class_idx in misclassified_images:
        if misclassified_images[class_idx]:
            plt.figure(figsize=(15, 3))
            plt.suptitle(f'Misclassified {class_names[class_idx]} Images')
            
            for i, img_data in enumerate(misclassified_images[class_idx][:num_misclassified]):
                plt.subplot(1, num_misclassified, i + 1)
                
                # Denormalize image
                img = img_data['image'].numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                plt.imshow(img)
                plt.axis('off')
                plt.title(f'True: {class_names[img_data["true"]]}\nPred: {class_names[img_data["pred"]]}')
            
            plt.tight_layout()
            plt.show()

