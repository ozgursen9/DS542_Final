#file for training the model

import os
import torch
from model import cnn_model
from data_loader import device, train_loader, valid_loader
from IPython.display import clear_output
import matplotlib.pyplot as plt
from model import save_model

loss_function = torch.nn.CrossEntropyLoss()

def quick_train_model(model_class, epochs=50, learning_rate=1e-3, **kwargs):
    os.makedirs('models', exist_ok=True)
    
    model = model_class(**kwargs)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []  # Added
    val_accuracies = []    # Renamed from accuracies
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    def plot_progress():
        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_title('Losses')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracies
        ax2.plot(train_accuracies, label='Train Accuracy')  # Added
        ax2.plot(val_accuracies, label='Val Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Acc: {train_accuracies[-1]:.1f}% | Val Acc: {val_accuracies[-1]:.1f}%')
        print(f'Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        correct = 0    # Added
        total = 0      # Added
        
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_function(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            # Calculate training accuracy
            _, predicted = outputs.max(1)
            total += batch_Y.size(0)
            correct += predicted.eq(batch_Y).sum().item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_Y in valid_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                outputs = model(batch_X)
                val_loss += loss_function(outputs, batch_Y).item()
                
                _, predicted = outputs.max(1)
                total += batch_Y.size(0)
                correct += predicted.eq(batch_Y).sum().item()
        
        avg_val_loss = val_loss / len(valid_loader)
        val_accuracy = 100. * correct / total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Early stopping
                # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            save_model(model, optimizer, 'models/best_model.pth')
            print(f"Model saved to models/best_model.pth")

        else:
            patience_counter += 1
        
        # Plot progress every epoch
        plot_progress()
        
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch + 1}')
            print(f'Best model was saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}')
            save_model(model, optimizer, 'models/final_model.pth')
            break
    
    if epoch == epochs - 1:  # If we completed all epochs
        print(f'\nTraining completed. Best model was saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}')
        save_model(model, optimizer, 'final_model.pth')

    return model
