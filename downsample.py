import cv2
import numpy as np
import os
import random

def get_distinct_colors(n):
    """Generates a list of distinct colors for different classes."""
    colors = []
    for i in range(n):
        # Generate random bright colors (BGR format for OpenCV)
        colors.append((random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))
    return colors

def draw_yolo_polygons(image_path, label_path, output_path, class_names=None, target_size=None):
    """
    Draws polygons on an image based on YOLO segmentation labels.
    
    Args:
        image_path (str): Path to source image.
        label_path (str): Path to YOLO format label file (txt).
        output_path (str): Path to save the visualized image.
        class_names (list): Optional list of class names.
        target_size (tuple): Optional (width, height) to resize the image to. 
                             Annotations will scale automatically.
    """
    # 1. Load Image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Resize image if target_size is specified
    if target_size is not None:
        # target_size should be (width, height)
        img = cv2.resize(img, target_size)

    h, w, _ = img.shape
    
    # Create a copy for the transparent overlay
    overlay = img.copy()
    
    # 2. Load Labels
    lines = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
    else:
        print(f"Warning: No label file found for {image_path}")

    # Generate colors for classes (assuming max 80 classes like COCO, or 10 for Dental)
    colors = get_distinct_colors(20)

    for line in lines:
        parts = list(map(float, line.strip().split()))
        class_id = int(parts[0])
        coords = parts[1:]
        
        # 3. Denormalize Coordinates (YOLO 0-1 to Pixels)
        # Reshape into a list of (x, y) points
        # Since 'w' and 'h' correspond to the (possibly resized) image dimensions,
        # this automatically handles the scaling of annotations.
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

parser = argparse.ArgumentParser(description="Visualize YOLO segmentation annotations")
parser.add_argument("--base-path", type=str, required=True, help="Base path to data directory")
args = parser.parse_args()

BASE_PATH = args.base_path
# Define your paths
img_dir = BASE_PATH + "images/train"
lbl_dir = BASE_PATH + "labels/train"
out_dir = BASE_PATH + "visualized_output/down"

# Ensure the output directory exists explicitly
os.makedirs(out_dir, exist_ok=True)

# Process specific files or all files in directory
import glob
import argparse
# image_files = glob.glob(os.path.join(img_dir, "*.jpg"))


# If you just want to test one file, uncomment this:
image_files = [BASE_PATH + "images/train/p002_F_52_003.jpg"]

for img_file in image_files:
    # Construct corresponding label path
    # Assumes labels are in a parallel folder with same name but .txt extension
    base_name = os.path.basename(img_file)
    label_name = os.path.splitext(base_name)[0] + ".txt"
    
    lbl_file = os.path.join(lbl_dir, label_name)
    
    out_file = BASE_PATH + "visualized_output/p002_F_52_003_downsampled.jpg"

    # Pass target_size to resize the output image and scale annotations
    draw_yolo_polygons(img_file, lbl_file, out_file, target_size=(640, 640))