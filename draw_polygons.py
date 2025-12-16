import cv2
import numpy as np
import os
import random
import glob
import sys

def get_distinct_colors(n):
    """Generates a list of distinct colors for different classes."""
    colors = []
    for i in range(n):
        # Generate random bright colors (BGR format for OpenCV)
        colors.append((random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))
    return colors

def draw_yolo_polygons(image_path, label_path, output_path, class_names=None):
    """
    Draws polygons on an image based on YOLO segmentation labels.
    """
    # 1. Load Image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    # Create a copy for the transparent overlay
    overlay = img.copy()
    
    # 2. Load Labels
    if not os.path.exists(label_path):
        print(f"Warning: No label file found for {image_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Generate colors for classes (assuming max 80 classes like COCO, or 10 for Dental)
    colors = get_distinct_colors(20)

    for line in lines:
        parts = list(map(float, line.strip().split()))
        class_id = int(parts[0])
        coords = parts[1:]
        
        # 3. Denormalize Coordinates (YOLO 0-1 to Pixels)
        # Reshape into a list of (x, y) points
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i+1] * h)
            points.append([x, y])
        
        pts_array = np.array(points, np.int32)
        pts_array = pts_array.reshape((-1, 1, 2))
        
        # Select color based on class_id
        color = colors[class_id % len(colors)]
        
        # 4. Draw Polygon (Filled + Outline)
        # Draw solid fill on the overlay
        cv2.fillPoly(overlay, [pts_array], color)
        
        # Draw outline on the main image (so edges are sharp)
        cv2.polylines(img, [pts_array], isClosed=True, color=color, thickness=2)

    # 5. Blend Overlay (Transparency)
    alpha = 0.4  # Transparency factor (0.0 - 1.0)
    img_result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # 6. Save Result
    # cv2.imwrite returns False if it fails, so we catch that here
    success = cv2.imwrite(output_path, img_result)
    
    if success:
        print(f"Saved visualization to: {os.path.abspath(output_path)}")
    else:
        print(f"ERROR: Could not save to {os.path.abspath(output_path)}")
        # Debugging hints
        if not os.path.exists(os.path.dirname(output_path)):
            print(f"  Reason: The output directory '{os.path.dirname(output_path)}' does not exist.")
        elif img_result is None or img_result.size == 0:
            print("  Reason: The image is empty.")

# --- Example Usage ---

if len(sys.argv) > 1:
    BASE_PATH = sys.argv[1]
else:
    BASE_PATH = "/users/PAS2699/brijeshnandaby/projects/cv/sahi_test/"
# draw_yolo_polygons(BASE_PATH + "images/train/p002_F_52_003.jpg", BASE_PATH + "labels/train/p002_F_52_003.txt", BASE_PATH + "visualized_output/p002_F_52_003.jpg")

# Define your paths
img_dir = BASE_PATH +"images"
lbl_dir = BASE_PATH + "labels"
out_dir = BASE_PATH + "visuals"

os.makedirs(out_dir, exist_ok=True)

# Process specific files or all files in directory
image_files = glob.glob(os.path.join(img_dir, "*.jpg"))

# If you just want to test one file, uncomment this:
# image_files = ["dataset/images/train/sample_001.jpg"]

for img_file in image_files:
    # Construct corresponding label path
    # Assumes labels are in a parallel folder with same name but .txt extension
    base_name = os.path.basename(img_file)
    label_name = os.path.splitext(base_name)[0] + ".txt"
    lbl_file = os.path.join(lbl_dir, label_name)
    
    out_file = os.path.join(out_dir, base_name.split('.')[0] + "_visualized.jpg")
    
    draw_yolo_polygons(img_file, lbl_file, out_file)