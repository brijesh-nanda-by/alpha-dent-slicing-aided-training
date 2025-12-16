import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import warnings

# Filter warnings
warnings.filterwarnings('ignore')

# ===============================================================================
# 1. SETUP & INSTALLATION
# ===============================================================================
print("Setting up environment...")
# Install SAHI if not present
os.system('pip install --quiet --upgrade ultralytics>=8.3.0 sahi shapely')

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image

# Configuration
# ---------------------------------------------------------
# UPDATE THIS PATH to where your best.pt is located
MODEL_WEIGHTS_PATH = sys.argv[1] if len(sys.argv) > 1 else 'results/test1/alphadent_yolov11x/weights/best.pt'

# Update this to your test images folder
TEST_IMAGES_PATH = sys.argv[2] if len(sys.argv) > 2 else '/users/PAS2699/brijeshnandaby/projects/cv/data/alpha-dent/AlphaDent/images/test'
# TEST_LABELS_PATH = 'data/alpha-dent/AlphaDent/labels/test'
# ---------------------------------------------------------

print(f"CUDA Available: {torch.cuda.is_available()}")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ===============================================================================
# 2. LOAD MODEL (NO TRAINING)
# ===============================================================================
print(f"\n=== Loading Model from {MODEL_WEIGHTS_PATH} ===")

if not os.path.exists(MODEL_WEIGHTS_PATH):
    raise FileNotFoundError(f"Could not find weights at {MODEL_WEIGHTS_PATH}. Please check the path.")

# Initialize SAHI Detection Model
# We use 'yolov8' model_type as the wrapper for v8/v11 is the same in SAHI
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=MODEL_WEIGHTS_PATH,
    confidence_threshold=0.10,  # Lower threshold to catch small caries
    device=device,
)

print("Model loaded successfully : " , detection_model.model.model.names)

# ===============================================================================
# 3. HELPER FUNCTIONS
# ===============================================================================
def mask_to_normalized_polygon(binary_mask, img_width, img_height, offset_x=0, offset_y=0):
    """Convert SAHI binary mask to normalized polygon string."""
    # Find contours
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
        
    # Get the largest contour
    cnt = max(contours, key=cv2.contourArea)
    
    if len(cnt) < 3: 
        return None
        
    # Flatten and Normalize
    polygon = []
    for point in cnt:
        x, y = point[0]
        # Apply offset if mask is cropped
        x_abs = x + offset_x
        y_abs = y + offset_y
        
        x_norm = max(0.0, min(1.0, float(x_abs) / img_width))
        y_norm = max(0.0, min(1.0, float(y_abs) / img_height))
        polygon.extend([x_norm, y_norm])
        
    return ' '.join([f'{coord:.6f}' for coord in polygon])

# ===============================================================================
# 4. RUN SAHI INFERENCE
# ===============================================================================
print("\n=== Running Sliced Inference (SAHI) ===")

test_images = sorted(glob.glob(f'{TEST_IMAGES_PATH}/*.jpg'))
if len(test_images) == 0:
    print(f"Error: No images found in {TEST_IMAGES_PATH}")
    sys.exit()

print(f"Processing {len(test_images)} test images...")
all_submission_rows = []

for i, image_path in enumerate(tqdm(test_images)):
    try:
        image_id = os.path.basename(image_path).replace('.jpg', '')
        image = read_image(image_path)
        height, width = image.shape[:2]
        
        # SAHI Sliced Prediction
        result = get_sliced_prediction(
            image,
            detection_model,
            slice_height=2500,  # Tile size
            slice_width=2500,
            overlap_height_ratio=0.25, # 25% overlap between tiles
            overlap_width_ratio=0.25,
            perform_standard_pred=False, # Set True if you have very large objects (e.g. whole jaw)
            verbose=0
        )
        
        # Process predictions
        for object_prediction in result.object_prediction_list:
            if object_prediction.mask:
                class_id = object_prediction.category.id
                confidence = object_prediction.score.value
                bool_mask = object_prediction.mask.bool_mask
                
                # Check if mask is full image or cropped to bbox
                offset_x, offset_y = 0, 0
                if bool_mask.shape[0] != height or bool_mask.shape[1] != width:
                    # If mask shape doesn't match image, assume it's relative to bbox
                    bbox = object_prediction.bbox
                    offset_x = bbox.min_x
                    offset_y = bbox.min_y
                
                poly_str = mask_to_normalized_polygon(bool_mask, width, height, offset_x, offset_y)
                
                if poly_str:
                    all_submission_rows.append({
                        'patient_id': image_id,
                        'class_id': int(class_id),
                        'confidence': float(confidence),
                        'poly': poly_str
                    })
                    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        continue

# ===============================================================================
# 5. CREATE SUBMISSION FILE
# ===============================================================================
print(f"\nGenerated {len(all_submission_rows)} predictions")

submission_df = pd.DataFrame(all_submission_rows)
all_test_ids = [os.path.basename(img).replace('.jpg', '') for img in test_images]

# Handle empty submission or missing images
if len(submission_df) > 0:
    predicted_ids = submission_df['patient_id'].unique()
    missing_ids = set(all_test_ids) - set(predicted_ids)
else:
    missing_ids = set(all_test_ids)
    submission_df = pd.DataFrame(columns=['patient_id', 'class_id', 'confidence', 'poly'])

# Add dummy predictions for missing images
if missing_ids:
    print(f"Adding dummy predictions for {len(missing_ids)} images")
    dummy_rows = []
    for img_id in missing_ids:
        dummy_rows.append({
            'patient_id': img_id,
            'class_id': 0,
            'confidence': 0.01,
            'poly': '0.1 0.1 0.1 0.2 0.2 0.2 0.2 0.1'
        })
    submission_df = pd.concat([submission_df, pd.DataFrame(dummy_rows)], ignore_index=True)

# Save Files
# 1. Standard Submission
submission_df = submission_df.sort_values(['patient_id', 'confidence'], ascending=[True, False])
submission_df = submission_df[['patient_id', 'class_id', 'confidence', 'poly']]
submission_df.to_csv('sahi_test_3000.csv', index=False)
print("Saved: sahi_test_3000.csv")

# 2. High Confidence Submission (Good for leaderboard probing)
high_conf_df = submission_df[submission_df['confidence'] >= 0.35].copy()
high_conf_ids = high_conf_df['patient_id'].unique()
missing_high = set(all_test_ids) - set(high_conf_ids)

# Ensure high conf submission is complete
if missing_high:
    for img_id in missing_high:
        img_preds = submission_df[submission_df['patient_id'] == img_id]
        if len(img_preds) > 0:
            high_conf_df = pd.concat([high_conf_df, img_preds.head(1)], ignore_index=True)
        else:
            dummy_row = pd.DataFrame([{
                'patient_id': img_id, 'class_id': 0, 'confidence': 0.01,
                'poly': '0.1 0.1 0.1 0.2 0.2 0.2 0.2 0.1'
            }])
            high_conf_df = pd.concat([high_conf_df, dummy_row], ignore_index=True)

high_conf_df.sort_values(['patient_id', 'confidence'], ascending=[True, False], inplace=True)
high_conf_df.to_csv('sahi_test_3000_high_conf.csv', index=False)
print("Saved: sahi_test_3000_high_conf.csv")

print("\nDone!")
