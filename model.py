#file for model architecture, saving and loading
import torch
import os
from data_loader import device

class cnn_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Reduce initial image size and use more efficient channel sizes
        self.features = torch.nn.Sequential(
            # First block: 256x256x3 -> 128x128x16
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, device=device),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            # Second block: 128x128x16 -> 32x32x32
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, device=device),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            # Third block: 32x32x32 -> 16x16x64
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, device=device),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            # Adaptive pooling to get fixed size regardless of input
            torch.nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
# Save the model
def save_model(model, optimizer, filename):
    """
    Save the model state dictionary, optimizer state, and architecture
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model and optimizer state dictionaries
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'architecture': type(model).__name__
    }, filename)  # Remove os.path.join('models', filename) since path already includes 'models/'
    print(f"Model saved to {filename}")

# Load the model
def load_model(filename):
    """
    Load the model state dictionary and architecture
    """
    try:
        # Load the saved state dictionary
        checkpoint = torch.load(os.path.join('models', filename))
        
        # Create a new model instance based on the saved architecture
        if checkpoint['architecture'] == 'cnn_model':
            model = cnn_model()
        else:
            raise ValueError(f"Unknown architecture: {checkpoint['architecture']}")
        
        # Load the state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print(f"Model loaded from models/{filename}")
        return model
    except FileNotFoundError:
        print(f"No saved model found at models/{filename}")
        return None
