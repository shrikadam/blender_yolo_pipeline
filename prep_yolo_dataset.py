import json
import os
import shutil
import random
import yaml

def prepare_yolo_dataset(
    coco_dir="output_dataset/coco_data", 
    output_dir="yolo_dataset", 
    train_ratio=0.8
):
    json_path = os.path.join(coco_dir, "coco_annotations.json")
    source_images_dir = os.path.join(coco_dir, "images")

    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return

    # 1. Read COCO data
    print("Loading COCO annotations...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # 2. Setup output directories
    dirs = {
        "train_img": os.path.join(output_dir, "images", "train"),
        "val_img": os.path.join(output_dir, "images", "val"),
        "train_lbl": os.path.join(output_dir, "labels", "train"),
        "val_lbl": os.path.join(output_dir, "labels", "val")
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 3. Map Categories (BlenderProc IDs 1, 2 -> YOLO IDs 0, 1)
    # This automatically fixes the 0-index requirement for YOLO
    cat_map = {}
    class_names = {}
    yolo_id = 0
    for cat in sorted(coco_data.get('categories', []), key=lambda x: x['id']):
        cat_map[cat['id']] = yolo_id
        class_names[yolo_id] = cat['name']
        yolo_id += 1

    # 4. Group annotations by image
    anns_by_img = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)

    # 5. Shuffle and split images
    images = coco_data.get('images', [])
    random.shuffle(images)
    
    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    print(f"Total images: {len(images)} | Training: {len(train_images)} | Validation: {len(val_images)}")

    # 6. Process and Copy Files
    def process_split(img_list, img_dest, lbl_dest):
        for img_info in img_list:
            # Clean filename
            file_name = img_info['file_name'].replace("images/", "").replace("images\\", "")
            src_img_path = os.path.join(source_images_dir, file_name)
            
            if not os.path.exists(src_img_path):
                continue

            # Copy Image
            shutil.copy(src_img_path, os.path.join(img_dest, file_name))

            # Create YOLO Label string
            img_w = img_info['width']
            img_h = img_info['height']
            label_lines = []
            
            for ann in anns_by_img.get(img_info['id'], []):
                # COCO: [x_min, y_min, width, height]
                x_min, y_min, bbox_w, bbox_h = ann['bbox']
                
                # YOLO: [class_id, x_center, y_center, norm_width, norm_height]
                x_center = (x_min + bbox_w / 2.0) / img_w
                y_center = (y_min + bbox_h / 2.0) / img_h
                norm_w = bbox_w / img_w
                norm_h = bbox_h / img_h
                
                mapped_cat = cat_map[ann['category_id']]
                label_lines.append(f"{mapped_cat} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

            # Write Label Text File
            txt_name = os.path.splitext(file_name)[0] + ".txt"
            with open(os.path.join(lbl_dest, txt_name), 'w') as txt_f:
                txt_f.write("\n".join(label_lines))

    print("Processing Training Set...")
    process_split(train_images, dirs["train_img"], dirs["train_lbl"])
    
    print("Processing Validation Set...")
    process_split(val_images, dirs["val_img"], dirs["val_lbl"])

    # 7. Generate dataset.yaml
    yaml_content = {
        "path": os.path.abspath(output_dir), # Absolute path is safest for YOLO
        "train": "images/train",
        "val": "images/val",
        "names": class_names
    }

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w') as yaml_f:
        yaml.dump(yaml_content, yaml_f, default_flow_style=False, sort_keys=False)

    print(f"\nSuccess! YOLO dataset created at: {os.path.abspath(output_dir)}")
    print(f"You can now run your training script pointing to: {yaml_path}")

if __name__ == "__main__":
    # Feel free to change the 0.8 to 0.9 if you want a 90/10 split instead
    prepare_yolo_dataset(coco_dir="output_dataset/coco_data", output_dir="yolo_dataset", train_ratio=0.8)