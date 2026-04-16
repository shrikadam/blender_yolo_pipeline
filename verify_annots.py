import json
import os
import random
import cv2

def verify_coco_dataset(dataset_dir="output_dataset/coco_data", num_samples=5):
    json_path = os.path.join(dataset_dir, "coco_annotations.json")
    images_dir = os.path.join(dataset_dir, "images")
    output_dir = "verification_samples"

    # 1. Load the COCO JSON
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directory for our visual checks
    os.makedirs(output_dir, exist_ok=True)

    # 2. Map Category IDs to Names (e.g., 0: "nic", 1: "sc")
    category_map = {}
    for cat in coco_data.get('categories', []):
        category_map[cat['id']] = cat['name']

    # 3. Pick random images
    images = coco_data.get('images', [])
    if len(images) == 0:
        print("No images found in the JSON.")
        return
        
    sampled_images = random.sample(images, min(num_samples, len(images)))

    # 4. Group annotations by image_id for fast lookup
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # 5. Process and draw each sampled image
    print(f"Drawing bounding boxes for {len(sampled_images)} random images...")
    
    for img_info in sampled_images:
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # In BlenderProc, file_name might include the "images/" prefix. Clean it up if needed.
        if file_name.startswith("images/"):
            file_name = file_name.replace("images/", "")
            
        img_path = os.path.join(images_dir, file_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found.")
            continue

        # Load image with OpenCV
        img = cv2.imread(img_path)
        
        # Get annotations for this image
        anns = annotations_by_image.get(img_id, [])
        
        for ann in anns:
            # COCO bbox format is [x_min, y_min, width, height]
            x, y, w, h = [int(v) for v in ann['bbox']]
            cat_id = ann['category_id']
            cat_name = category_map.get(cat_id, f"Class_{cat_id}")
            
            # Choose a color based on category (NIC: Green, SC: Blue)
            color = (0, 255, 0) if "nic" in cat_name.lower() else (255, 0, 0)
            
            # Draw Bounding Box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Draw Label Background (for text readability)
            label = f"{cat_name}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y), color, -1)
            
            # Draw Text
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save the drawn image
        out_path = os.path.join(output_dir, f"verified_{file_name}")
        cv2.imwrite(out_path, img)
        print(f"Saved {out_path} with {len(anns)} detections.")

    print(f"\nVerification complete! Check the '{output_dir}' folder.")

if __name__ == "__main__":
    verify_coco_dataset()