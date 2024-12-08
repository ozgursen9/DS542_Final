# DS542_Final Project

This repository contains the code and resources for the DS542 Final Project. The goal of this project is to process and analyze car image data using convolutional neural networks (CNNs) to predict damage.

## Data
The data used in this project is available in the following Google Drive folder: [Project Data](https://drive.google.com/drive/folders/1znDzs-fC9ysmS673rcSSWKB2hgRzb7Qq?usp=drive_link)

## Repository Structure
### **Image Processing**
- **`image_processing/`**: Contains scripts and steps to preprocess and process the images used throughout the project.

### **Models**
- **`models/`**: 
  - `best_model`: Stores the model with the lowest validation loss.
  - `final_model`: Contains the model saved after completing training.
  
### **Scripts**
1. **`data_loader.py`**: 
   - Code to load the image data.
   - Update the `DATA_DIR` variable to specify the directory containing your data.
   
2. **`train_model.py`**: 
   - Script for training the CNN model.
   - Implements early stopping: Training stops if the current validation loss exceeds the average validation loss.
   
3. **`model.py`**: 
   - Defines the CNN architecture used in this project.
   
4. **`evaluation.py`**: 
   - Provides basic evaluation metrics and generates a confusion matrix for model predictions.
   
5. **`analyze_predictions.py`**: 
   - Utilizes Grad-CAM to highlight areas of the image that significantly influence the model's decision-making.

### **Main Folders**
- **`main_baseline/`**: Model trained on raw images with minimal preprocessing (license plate masking only).
- **`main_segmented/`**: Model trained on segmented images.
- **`main_reduced_baseline/`**: Model trained on a subset of raw images with high-quality segmentation. Preprocessing includes only license plate masking.
- **`main_reduced_segmented/`**: Model trained on a subset of segmented images, selected based on the following quality criteria:
  - Percentage of black pixels in the image is less than 80%.

