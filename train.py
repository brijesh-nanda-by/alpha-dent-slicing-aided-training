import os
import sys
import time
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
import yaml
import random
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'
# Fix for CUDA OOM fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Install required packages
print("Installing required packages...")
# Using --quiet (shorthand -q) to suppress installation output for a cleaner log
os.system('pip install --quiet --upgrade ultralytics>=8.3.0')

import torch
from ultralytics import YOLO
import ultralytics
print(f"Ultralytics Version: {ultralytics.__version__}")

# Set deterministic behavior for PyTorch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f'\nPyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Device: {torch.cuda.get_device_name(0)}')

# Define paths
BASE_PATH = sys.argv[1] if len(sys.argv) > 1 else '/home/bbrundayogananda/projects/cv/data640'
TRAIN_IMAGES_PATH = f'{BASE_PATH}/images/train'
VALID_IMAGES_PATH = f'{BASE_PATH}/images/valid'
TEST_IMAGES_PATH = f'{BASE_PATH}/images/test'
TRAIN_LABELS_PATH = f'{BASE_PATH}/labels/train'
VALID_LABELS_PATH = f'{BASE_PATH}/labels/valid'
# Output paths
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else '/home/bbrundayogananda/projects/cv/results/testv11'
WEIGHTS_DIR = f'{OUTPUT_DIR}/weights'
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Define class information
CLASS_INFO = {
    0: {'name': 'Abrasion', 'description': 'Teeth with mechanical wear of hard tissues'},
    1: {'name': 'Filling', 'description': 'Dental fillings of various types'},
    2: {'name': 'Crown', 'description': 'Dental crown (restoration)'},
    3: {'name': 'Caries Class 1', 'description': 'Caries in fissures and pits'},
    4: {'name': 'Caries Class 2', 'description': 'Caries on proximal surfaces of molars/premolars'},
    5: {'name': 'Caries Class 3', 'description': 'Caries on proximal surfaces of incisors/canines without incisal edge'},
    6: {'name': 'Caries Class 4', 'description': 'Caries on proximal surfaces of incisors/canines with incisal edge'},
    7: {'name': 'Caries Class 5', 'description': 'Cervical caries (buccal/lingual surfaces)'},
    8: {'name': 'Caries Class 6', 'description': 'Caries on incisal edges or cusps'}
}

# Create YAML configuration for YOLO
print("\n=== Creating YOLO Configuration ===")
yolo_config = {
    'path': BASE_PATH,
    'train': 'images/train',
    'val': 'images/valid',
    'test': 'images/test',
    'nc': 9,
    'names': [CLASS_INFO[i]['name'] for i in range(9)]
}

# Save the configuration
CUSTOM_YAML_PATH = f'{OUTPUT_DIR}/alphadent_config.yaml'
with open(CUSTOM_YAML_PATH, 'w') as f:
    yaml.dump(yolo_config, f, default_flow_style=False)
print(f"Created custom YAML config at: {CUSTOM_YAML_PATH}")

# Count images
train_images = sorted(glob.glob(f'{TRAIN_IMAGES_PATH}/*.jpg'))
valid_images = sorted(glob.glob(f'{VALID_IMAGES_PATH}/*.jpg'))
test_images = sorted(glob.glob(f'{TEST_IMAGES_PATH}/*.jpg'))

print(f"\n=== Dataset Statistics ===")
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(valid_images)}")
print(f"Test images: {len(test_images)}")

# Analyze class distribution
def analyze_class_distribution(labels_path):
    """Analyze class distribution in dataset."""
    class_counts = {i: 0 for i in range(9)}
    total_annotations = 0
    
    label_files = glob.glob(f'{labels_path}/*.txt')
    
    for label_file in tqdm(label_files, desc="Analyzing labels", leave=False):
        if os.path.exists(label_file) and os.path.getsize(label_file) > 0:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip():
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if 0 <= class_id < 9:
                                    class_counts[class_id] += 1
                                    total_annotations += 1
            except Exception:
                continue
    
    return class_counts, total_annotations

print("\n=== Analyzing Class Distribution ===")
train_class_counts, train_total = analyze_class_distribution(TRAIN_LABELS_PATH)
valid_class_counts, valid_total = analyze_class_distribution(VALID_LABELS_PATH)

print(f"Training set: {train_total} total annotations")
print(f"Validation set: {valid_total} total annotations")

# Training configuration
print("\n=== Model Training Configuration ===")
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 35
IMAGE_SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 640
# Reduced batch size to avoid CUDA OOM with YOLOv11x
BATCH_SIZE = 8 if torch.cuda.is_available() else 2
PATIENCE = 5

print(f"Epochs: {EPOCHS}")
print(f"Image Size: {IMAGE_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Early Stopping Patience: {PATIENCE}")

# Initialize and train model
print("\n=== Starting Model Training ===")
model = YOLO('yolo11x-seg.pt')

# Train the model with optimized parameters
""" results = model.train(
    data=CUSTOM_YAML_PATH,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    patience=PATIENCE,
    save=True,
    save_period=10,
    project=OUTPUT_DIR,
    name='alphadent_yolov11x',
    exist_ok=True,
    pretrained=True,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    plots=True,
    device=0 if torch.cuda.is_available() else 'cpu',
    workers=2,
    verbose=True,
    amp=True,
    val=True
) """

print("\nTraining completed!")

# Load best model
print("\n=== Loading Best Model ===")
best_model_path = f'{OUTPUT_DIR}/alphadent_yolov11x/weights/best.pt'
if os.path.exists(best_model_path):
    model = YOLO(best_model_path)
    print(f"Loaded best model from: {best_model_path}")
else:
    last_model_path = f'{OUTPUT_DIR}/alphadent_yolov11x/weights/last.pt'
    if os.path.exists(last_model_path):
        model = YOLO(last_model_path)
        print(f"Loaded last model from: {last_model_path}")
    else:
        print("Warning: No trained model found, using pretrained model")
        model = YOLO('weights/yolo11x-seg.pt')

# Validate model
print("\n=== Model Validation ===")
try:
    metrics = model.val(
        data=CUSTOM_YAML_PATH,
        imgsz=IMAGE_SIZE,
        batch=1,
        conf=0.001,
        iou=0.5,
        max_det=300,
        device=0 if torch.cuda.is_available() else 'cpu',
        plots=False,
        save_json=False,
    )
    
    print(f"\nValidation Results:")
    print(f"mAP@50: {metrics.seg.map50:.4f}")
    print(f"mAP@50-95: {metrics.seg.map:.4f}")
except Exception as e:
    print(f"Validation error (non-critical): {e}")

# Inference on test set
print("\n=== Running Inference on Test Set ===")

def convert_to_submission_format(results, image_paths):
    """Convert YOLO results to competition submission format."""
    submission_rows = []
    
    for idx, result in enumerate(results):
        # Get image ID (filename without extension)
        image_id = os.path.basename(image_paths[idx]).replace('.jpg', '')
        
        if result.masks is not None and len(result.masks) > 0:
            try:
                # Get masks, classes, and confidences
                masks = result.masks.xy
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                h, w = result.orig_shape
                
                # Process each detection
                for mask_idx in range(len(masks)):
                    if mask_idx < len(classes) and mask_idx < len(confidences):
                        polygon = masks[mask_idx]
                        
                        if len(polygon) >= 3:  # Valid polygon (at least 3 points)
                            # Normalize coordinates to [0, 1]
                            normalized_coords = []
                            for point in polygon:
                                x_norm = float(point[0]) / w
                                y_norm = float(point[1]) / h
                                # Ensure coordinates are within [0, 1]
                                x_norm = max(0.0, min(1.0, x_norm))
                                y_norm = max(0.0, min(1.0, y_norm))
                                normalized_coords.extend([x_norm, y_norm])
                            
                            # Format polygon string
                            poly_str = ' '.join([f'{coord:.6f}' for coord in normalized_coords])
                            
                            submission_rows.append({
                                'patient_id': image_id,
                                'class_id': int(classes[mask_idx]),
                                'confidence': float(confidences[mask_idx]),
                                'poly': poly_str
                            })
            except Exception as e:
                print(f"Error processing result for image {idx}: {e}")
                continue
    
    return submission_rows

# Process test images
test_images = sorted(glob.glob(f'{TEST_IMAGES_PATH}/*.jpg'))
all_submission_rows = []
INFERENCE_BATCH_SIZE = 8 if torch.cuda.is_available() else 4

print(f"Processing {len(test_images)} test images...")

# Process in batches
for i in tqdm(range(0, len(test_images), INFERENCE_BATCH_SIZE)):
    batch_images = test_images[i:i + INFERENCE_BATCH_SIZE]
    
    try:
        # Run inference
        results = model.predict(
            batch_images,
            imgsz=IMAGE_SIZE,
            conf=0.20,  # Confidence threshold
            iou=0.50,   # NMS IoU threshold
            max_det=300,
            device=0 if torch.cuda.is_available() else 'cpu',
            verbose=False,
            agnostic_nms=True,
            retina_masks=True,
        )
        
        # Convert results to submission format
        batch_rows = convert_to_submission_format(results, batch_images)
        all_submission_rows.extend(batch_rows)
        
    except Exception as e:
        print(f"Error in batch {i//INFERENCE_BATCH_SIZE}: {e}")
        continue

print(f"\nGenerated {len(all_submission_rows)} predictions")

# Create submission DataFrame
print("\n=== Creating Submission File ===")
submission_df = pd.DataFrame(all_submission_rows)

# Ensure all test images have at least one prediction
all_test_ids = [os.path.basename(img).replace('.jpg', '') for img in test_images]
if len(submission_df) > 0:
    predicted_ids = submission_df['patient_id'].unique()
    missing_ids = set(all_test_ids) - set(predicted_ids)
else:
    missing_ids = set(all_test_ids)

# Add dummy predictions for images without detections
if missing_ids:
    print(f"Adding dummy predictions for {len(missing_ids)} images without detections")
    dummy_rows = []
    for img_id in missing_ids:
        # Create a small dummy polygon
        dummy_rows.append({
            'patient_id': img_id,
            'class_id': 0,  # Default to class 0 (Abrasion)
            'confidence': 0.01,  # Very low confidence
            'poly': '0.1 0.1 0.1 0.2 0.2 0.2 0.2 0.1'  # Small square polygon
        })
    
    submission_df = pd.concat([submission_df, pd.DataFrame(dummy_rows)], ignore_index=True)

# Sort by patient_id and then by confidence (descending)
submission_df = submission_df.sort_values(['patient_id', 'confidence'], ascending=[True, False])

# Ensure correct column order
submission_df = submission_df[['patient_id', 'class_id', 'confidence', 'poly']]

# Save the main submission file
submission_df.to_csv('submission.csv', index=False)
print("Main submission file created: submission.csv")

# Verify submission format
print("\n=== Verifying Submission Format ===")
print(f"Total predictions: {len(submission_df)}")
print(f"Unique images: {submission_df['patient_id'].nunique()}")
print(f"All test images included: {submission_df['patient_id'].nunique() == len(test_images)}")

# Display first few rows
print("\nFirst 5 rows of submission:")
print(submission_df.head())

# Check for any potential issues
print("\n=== Checking for Potential Issues ===")

# Check for missing test images
missing_in_submission = set(all_test_ids) - set(submission_df['patient_id'].unique())
if missing_in_submission:
    print(f"WARNING: Missing images in submission: {missing_in_submission}")
else:
    print("✓ All test images have predictions")

# Check class distribution
print("\nPredictions per class:")
class_dist = submission_df['class_id'].value_counts().sort_index()
for class_id, count in class_dist.items():
    if 0 <= class_id < 9:
        print(f"  Class {class_id} ({CLASS_INFO[class_id]['name']}): {count}")

# Check confidence distribution
print(f"\nConfidence statistics:")
print(f"  Min: {submission_df['confidence'].min():.4f}")
print(f"  Max: {submission_df['confidence'].max():.4f}")
print(f"  Mean: {submission_df['confidence'].mean():.4f}")
print(f"  Median: {submission_df['confidence'].median():.4f}")

# Create alternative submission with higher confidence threshold
print("\n=== Creating Alternative Submission (Higher Confidence) ===")
high_conf_df = submission_df[submission_df['confidence'] >= 0.3].copy()

# Ensure all images still have at least one prediction
high_conf_ids = high_conf_df['patient_id'].unique()
missing_high_conf = set(all_test_ids) - set(high_conf_ids)

if missing_high_conf:
    # Add the highest confidence prediction for each missing image
    for img_id in missing_high_conf:
        img_preds = submission_df[submission_df['patient_id'] == img_id]
        if len(img_preds) > 0:
            # Add the highest confidence prediction
            high_conf_df = pd.concat([high_conf_df, img_preds.head(1)], ignore_index=True)
        else:
            # Add dummy prediction
            dummy_row = pd.DataFrame([{
                'patient_id': img_id,
                'class_id': 0,
                'confidence': 0.01,
                'poly': '0.1 0.1 0.1 0.2 0.2 0.2 0.2 0.1'
            }])
            high_conf_df = pd.concat([high_conf_df, dummy_row], ignore_index=True)

high_conf_df = high_conf_df.sort_values(['patient_id', 'confidence'], ascending=[True, False])
high_conf_df.to_csv('submission_high_conf.csv', index=False)
print(f"Created high confidence submission with {len(high_conf_df)} predictions")

print("\n=== Pipeline Completed Successfully! ===")
print("Submission files created:")
print("  - submission.csv (main submission)")
print("  - submission_high_conf.csv (alternative with higher confidence threshold)")