import os
import yaml
import cv2
import shutil
import glob
import random
import albumentations as A
import argparse

# --- Default Configuration ---
DATA_ROOT = '../data'
DATA_YAML_PATH = os.path.join(DATA_ROOT, 'data.yaml')
TEMP_IMG_DIR = os.path.join(DATA_ROOT, 'images_all_temp') # Temp dir for all aggregated images
TEMP_LBL_DIR = os.path.join(DATA_ROOT, 'labels_all_temp') # Temp dir for all aggregated labels

# --- Labeling Tool Related (Embedded annotation.py logic) ---
points = []
current_boxes = []
img_display = None
img_raw = None
img_h, img_w = 0, 0

def convert_to_yolo(p1, p2, w, h):
    x1, y1 = p1
    x2, y2 = p2
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    box_w, box_h = max_x - min_x, max_y - min_y
    center_x = min_x + box_w / 2.0
    center_y = min_y + box_h / 2.0
    return (center_x / w, center_y / h, box_w / w, box_h / h)

def draw_boxes(img, boxes):
    for box in boxes:
        cx, cy, bw, bh = box
        pixel_w = int(bw * img_w)
        pixel_h = int(bh * img_h)
        pixel_cx = int(cx * img_w)
        pixel_cy = int(cy * img_h)
        x1 = int(pixel_cx - pixel_w / 2)
        y1 = int(pixel_cy - pixel_h / 2)
        cv2.rectangle(img, (x1, y1), (x1 + pixel_w, y1 + pixel_h), (0, 0, 255), 2)

def mouse_callback(event, x, y, flags, param):
    global points, current_boxes, img_display, img_raw
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('Labeling Tool', img_display)
        if len(points) == 2:
            yolo_box = convert_to_yolo(points[0], points[1], img_w, img_h)
            current_boxes.append(yolo_box)
            img_copy = img_raw.copy()
            draw_boxes(img_copy, current_boxes)
            img_display = img_copy
            cv2.imshow('Labeling Tool', img_display)
            points = []

def run_labeling(raw_folder, class_id, class_name, start_index=0):
    global img_raw, img_display, img_h, img_w, current_boxes, points
    
    files = [f for f in os.listdir(raw_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    files.sort()
    
    print(f"\n--- Start Labeling Class: {class_name} (ID: {class_id}) ---")
    print(f"Source Folder: {raw_folder}")
    print(f"Found {len(files)} images to process")
    
    local_index = start_index

    cv2.namedWindow('Labeling Tool')
    cv2.setMouseCallback('Labeling Tool', mouse_callback)

    for filename in files:
        # Define target filename
        new_name = f"{class_name}_{local_index:05d}"
        dst_img_path = os.path.join(TEMP_IMG_DIR, new_name + ".png")
        dst_lbl_path = os.path.join(TEMP_LBL_DIR, new_name + ".txt")

        # If target file exists, skip (resume labeling)
        if os.path.exists(dst_img_path) and os.path.exists(dst_lbl_path):
            print(f"Skipping {filename}, already labeled as {new_name}")
            local_index += 1
            continue

        file_path = os.path.join(raw_folder, filename)
        img_raw = cv2.imread(file_path)
        if img_raw is None: continue
            
        img_h, img_w = img_raw.shape[:2]
        img_display = img_raw.copy()
        current_boxes = []
        points = []
        
        saved = False
        while True:
            cv2.imshow('Labeling Tool', img_display)
            key = cv2.waitKey(20) & 0xFF
            
            if key == 13: # Enter
                if not current_boxes:
                    print("Warning: No boxes labeled for current image. Skip saving? (Press Enter to continue to next, unsaved)")
                    break
                
                # Copy Image
                cv2.imwrite(dst_img_path, img_raw) # Convert to PNG

                
                # Save Label
                with open(dst_lbl_path, 'w') as f:
                    for box in current_boxes:
                        line = f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"
                        f.write(line)
                print(f"Saved: {new_name}")
                local_index += 1
                saved = True
                break
            elif key == ord('c'): # Clear
                current_boxes = []
                points = []
                img_display = img_raw.copy()
            elif key == ord('q'): # Quit
                cv2.destroyAllWindows()
                return False # Stop pipeline
            elif key == ord('s'): # Skip
                print(f"Skipped: {filename}")
                break
        
        if key == ord('q'): return False

    cv2.destroyAllWindows()
    return True

# --- Data Augmentation ---
def run_augmentation_in_folder(img_dir, lbl_dir, aug_ratio):
    if aug_ratio <= 0: return

    # Augmentation Pipeline
    transform = A.Compose([
        A.SafeRotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            A.ElasticTransform(alpha=1, sigma=50, p=1),
            A.OpticalDistortion(distort_limit=0.1, p=1),
        ], p=0.3),
        A.OneOf([
            A.Blur(blur_limit=3, p=1),
            A.GaussianBlur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ], p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HorizontalFlip(p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))

    img_list = glob.glob(os.path.join(img_dir, "*.png"))
    
    for img_path in img_list:
        # Only augment "original" images (prevent re-augmenting, check via filename)
        # Simple check: skip if filename contains "_aug_"
        if "_aug_" in img_path: continue

        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        lbl_path = os.path.join(lbl_dir, name + ".txt")
        
        image = cv2.imread(img_path)
        if image is None or not os.path.exists(lbl_path): continue

        bboxes = []
        class_labels = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append(list(map(float, parts[1:5])))
        
        for i in range(int(aug_ratio)):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_lbls = augmented['class_labels']
                
                if len(aug_bboxes) == 0: continue # Don't save if boxes disappeared

                new_name = f"{name}_aug_{i}"
                cv2.imwrite(os.path.join(img_dir, new_name + ".png"), aug_img)
                
                with open(os.path.join(lbl_dir, new_name + ".txt"), 'w') as f_out:
                    for bbox, cls_id in zip(aug_bboxes, aug_lbls):
                        xc, yc, w, h = bbox
                        xc, yc = min(max(xc, 0), 1), min(max(yc, 0), 1)
                        w, h = min(max(w, 0), 1), min(max(h, 0), 1)
                        f_out.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            except Exception as e:
                pass # Ignore errors


# --- Dataset Splitting ---
def run_split(train_r, val_r, test_r):
    print(f"\n--- Start Splitting Dataset ({train_r}:{val_r}:{test_r}) ---")
    
    subsets = ['train', 'valid', 'test']
    # Clear old split folders but keep structure
    for s in subsets:
        shutil.rmtree(os.path.join(DATA_ROOT, s), ignore_errors=True)
        os.makedirs(os.path.join(DATA_ROOT, s, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DATA_ROOT, s, 'labels'), exist_ok=True)

    all_files = [f for f in os.listdir(TEMP_IMG_DIR) if f.endswith('.png')]
    random.shuffle(all_files)
    
    total = len(all_files)
    n_train = int(total * train_r)
    n_val = int(total * val_r)
    
    splits = {
        'train': all_files[:n_train],
        'valid': all_files[n_train:n_train+n_val],
        'test': all_files[n_train+n_val:]
    }

    for subset, files in splits.items():
        dst_img_dir = os.path.join(DATA_ROOT, subset, 'images')
        dst_lbl_dir = os.path.join(DATA_ROOT, subset, 'labels')
        
        for f_name in files:
            name = os.path.splitext(f_name)[0]
            # Move Image
            shutil.copy(os.path.join(TEMP_IMG_DIR, f_name), os.path.join(dst_img_dir, f_name))
            # Move Label
            lbl_src = os.path.join(TEMP_LBL_DIR, name + ".txt")
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, os.path.join(dst_lbl_dir, name + ".txt"))
        
        print(f"  -> {subset}: {len(files)} 张")

def main():
    parser = argparse.ArgumentParser(description="Auto Data Pipeline")
    parser.add_argument('--aug_ratio', type=int, default=0, help='Augmentation ratio per image (0 to disable)')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--skip_labeling', action='store_true', help='Skip manual labeling step')
    args = parser.parse_args()

    # 1. Parse YAML
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Error: {DATA_YAML_PATH} not found!")
        return
    
    with open(DATA_YAML_PATH, 'r') as f:
        data_cfg = yaml.safe_load(f)
        class_names = data_cfg.get('names', [])
        print(f"Detected Classes from YAML: {class_names}")

    # 2. Prepare Temp Aggregation Dirs
    os.makedirs(TEMP_IMG_DIR, exist_ok=True)
    os.makedirs(TEMP_LBL_DIR, exist_ok=True)

    # 3. Iterate Classes for Labeling
    if not args.skip_labeling:
        for idx, cls_name in enumerate(class_names):
            raw_dir = os.path.join(DATA_ROOT, cls_name)
            if not os.path.exists(raw_dir):
                print(f"Warning: Source folder {raw_dir} for '{cls_name}' not found, skipping.")
                continue
            
            # Find max index for this class to resume numbering (e.g., bottles_00123)
            # Simple strategy: always rescan
            success = run_labeling(raw_dir, idx, cls_name, start_index=0)
            if not success:
                print("Program interrupted by user.")
                return
    else:
        print("Skipping labeling step...")

    # 4. Split Dataset (Split original data first)
    run_split(args.train_ratio, args.val_ratio, args.test_ratio)

    # 5. Data Augmentation (Augment separately for each subset after splitting)
    if args.aug_ratio > 0:
        print(f"\n--- Start Data Augmentation (Ratio: {args.aug_ratio}) ---")
        # Iterate train, valid, test folders
        for subset in ['train', 'valid', 'test']:
            subset_img_dir = os.path.join(DATA_ROOT, subset, 'images')
            subset_lbl_dir = os.path.join(DATA_ROOT, subset, 'labels')
            
            if os.path.exists(subset_img_dir):
                print(f"  Processing subset: {subset}")
                run_augmentation_in_folder(subset_img_dir, subset_lbl_dir, args.aug_ratio)
    else:
        print("\n--- Skip Augmentation (Ratio=0) ---")

    print("\n=== Pipeline Completed Successfully! ===")
    print(f"Data is ready in: {DATA_ROOT}/[train|valid|test]")

    # Optional: cleanup temp dirs
    shutil.rmtree(TEMP_IMG_DIR)
    shutil.rmtree(TEMP_LBL_DIR)

if __name__ == "__main__":
    main()