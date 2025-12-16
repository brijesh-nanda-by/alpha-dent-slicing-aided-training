import os
import cv2
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon, box
from shapely.validation import make_valid
from shapely.affinity import translate
import glob
import sys

# --- Helper Functions ---

def read_yolo_segmentation(line, img_w, img_h):
    parts = list(map(float, line.strip().split()))
    class_id = int(parts[0])
    coords = parts[1:]
    points = []
    for i in range(0, len(coords), 2):
        x = coords[i] * img_w
        y = coords[i+1] * img_h
        points.append((x, y))
    return class_id, points

def format_yolo_segmentation(class_id, polygon, tile_w, tile_h):
    if polygon.is_empty: return None
    
    # Handle MultiPolygons by taking the largest component
    # (YOLO format usually expects one closed loop per line)
    if polygon.geom_type == 'MultiPolygon':
        if not polygon.geoms: return None
        polygon = max(polygon.geoms, key=lambda a: a.area)

    try:
        x, y = polygon.exterior.coords.xy
    except AttributeError:
        # Failsafe if we somehow still got a non-polygon
        return None

    coords_str = []
    for i in range(len(x)):
        nx = max(0.0, min(1.0, x[i] / tile_w))
        ny = max(0.0, min(1.0, y[i] / tile_h))
        coords_str.append(f"{nx:.6f} {ny:.6f}")
        
    return f"{class_id} " + " ".join(coords_str)

# --- Main Logic ---

def generate_multiset_crops(
    image_path, label_path, output_img_dir, output_lbl_dir, 
    tile_w=640, tile_h=640, 
    offsets=[(0, 0)], 
    min_area_ratio=0.1
):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None: 
        # print(f"Warning: Could not read image {image_path}")
        return
    img_h_real, img_w_real, _ = img.shape
    
    # 2. Load Polygons
    polygons = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                try:
                    cid, points = read_yolo_segmentation(line, img_w_real, img_h_real)
                    if len(points) >= 3:
                        poly = Polygon(points)
                        if not poly.is_valid:
                            poly = make_valid(poly)
                        polygons.append((cid, poly))
                except Exception as e:
                    print(f"Skipping bad line in {label_path}: {e}")

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 3. Iterate through offsets (Passes)
    for pass_idx, (off_x, off_y) in enumerate(offsets):
        
        stride_x = tile_w 
        stride_y = tile_h 

        for y in range(off_y, img_h_real, stride_y):
            for x in range(off_x, img_w_real, stride_x):
                
                # Define Crop Coordinates
                x_end = min(x + tile_w, img_w_real)
                y_end = min(y + tile_h, img_h_real)
                x_start = x_end - tile_w
                y_start = y_end - tile_h
                
                # Handle small images by clamping start to 0
                if x_start < 0: x_start = 0
                if y_start < 0: y_start = 0

                # Create Shapely Box
                tile_box = box(x_start, y_start, x_end, y_end)
                
                tile_annotations = []
                
                for cid, poly in polygons:
                    if not tile_box.intersects(poly):
                        continue
                        
                    intersection = tile_box.intersection(poly)
                    
                    if intersection.is_empty:
                        continue

                    # --- FIXED LOGIC START ---
                    # Explicitly filter for Polygons only.
                    # This prevents LineStrings (edges) or Points from crashing the script.
                    valid_parts = []
                    
                    if intersection.geom_type == 'Polygon':
                        valid_parts.append(intersection)
                    elif intersection.geom_type == 'MultiPolygon':
                        valid_parts.extend(intersection.geoms)
                    elif intersection.geom_type == 'GeometryCollection':
                        # A messy intersection might return a mix of lines and polygons
                        for geom in intersection.geoms:
                            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                                if geom.geom_type == 'MultiPolygon':
                                    valid_parts.extend(geom.geoms)
                                else:
                                    valid_parts.append(geom)
                    # Note: LineStrings and Points are explicitly ignored here.
                    # --- FIXED LOGIC END ---

                    for part in valid_parts:
                        # Filter tiny slivers (< 0.5% of tile area)
                        if part.area < (tile_w * tile_h * 0.005):
                            continue
                            
                        shifted_poly = translate(part, xoff=-x_start, yoff=-y_start)
                        
                        yolo_line = format_yolo_segmentation(cid, shifted_poly, tile_w, tile_h)
                        if yolo_line:
                            tile_annotations.append(yolo_line)

                # Save Tile
                if tile_annotations:
                    tile_name = f"{base_name}_pass{pass_idx}_x{x_start}_y{y_start}"
                    
                    crop_img = img[y_start:y_end, x_start:x_end]
                    
                    if crop_img.size == 0: continue

                    # Pad if necessary to match tile_w/tile_h
                    if crop_img.shape[0] != tile_h or crop_img.shape[1] != tile_w:
                        pad_h = tile_h - crop_img.shape[0]
                        pad_w = tile_w - crop_img.shape[1]
                        crop_img = cv2.copyMakeBorder(crop_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))

                    cv2.imwrite(os.path.join(output_img_dir, f"{tile_name}.jpg"), crop_img)
                    with open(os.path.join(output_lbl_dir, f"{tile_name}.txt"), 'w') as f:
                        f.write("\n".join(tile_annotations))

# --- Execution Block ---

W, H = 2560, 2560

OFFSET_SETS = [
    (0, 0),         
    (int(W/2), int(H/2)), 
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, "../data/alpha-dent/AlphaDent")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        BASE_PATH = sys.argv[1]

    SPLITS = ['train', 'valid', 'test']
    
    for split in SPLITS:
        print(f"\nProcessing split: {split}")
        input_img_dir = os.path.join(BASE_PATH, "images", split)
        
        # Skip if split directory doesn't exist
        if not os.path.exists(input_img_dir):
            print(f"  Skipping {split} (directory not found: {input_img_dir})")
            continue
            
        out_img_dir = os.path.join(BASE_PATH, "cropped_images", f"{split}2560")
        out_lbl_dir = os.path.join(BASE_PATH, "cropped_labels", f"{split}2560")

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        # Support multiple extensions
        all_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            all_images.extend(glob.glob(os.path.join(input_img_dir, ext)))
        
        # Using tqdm to show progress
        for img_path in tqdm(all_images, desc=f"Cropping {split}"):
            # Infer label path by replacing 'images' with 'labels' and extension with .txt
            # This handles different extensions robustly
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # Assuming standard YOLO structure: ../images/split/img.jpg -> ../labels/split/img.txt
            label_dir = os.path.dirname(img_path).replace("images", "labels")
            label_path = os.path.join(label_dir, base_name + ".txt")
            
            generate_multiset_crops(
                img_path, label_path, 
                out_img_dir, out_lbl_dir, 
                tile_w=W, tile_h=H, 
                offsets=OFFSET_SETS
            )