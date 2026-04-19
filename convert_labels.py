import json
import os
from tqdm import tqdm

# =========================
# PATHS (EDIT THESE IF NEEDED)
# =========================

# FLIR original JSON files
TRAIN_JSON = r"D:\AI project\FLIR\train\thermal_annotations.json"
VAL_JSON   = r"D:\AI project\FLIR\val\thermal_annotations.json"

# Your working dataset images
TRAIN_IMG_DIR = r"D:\AI project\dataset\images\train"
VAL_IMG_DIR   = r"D:\AI project\dataset\images\val"

# Output YOLO labels
TRAIN_LABEL_DIR = r"D:\AI project\dataset\labels\train"
VAL_LABEL_DIR   = r"D:\AI project\dataset\labels\val"


# =========================
# CONVERT FUNCTION
# =========================

def convert(json_path, image_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)

    print(f"\nProcessing: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    annotations = data["annotations"]

    # Map annotations to each image
    ann_map = {}
    for ann in annotations:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    count = 0

    for img_id, img in tqdm(images.items()):
        # 🔥 IMPORTANT FIX (remove folder prefix)
        filename = os.path.basename(img["file_name"])

        img_path = os.path.join(image_dir, filename)

        # Skip if image not present
        if not os.path.exists(img_path):
            continue

        h, w = img["height"], img["width"]
        label_path = os.path.join(label_dir, filename.replace(".jpeg", ".txt"))

        with open(label_path, "w") as f:
            for ann in ann_map.get(img_id, []):
                x, y, bw, bh = ann["bbox"]

                # Convert COCO → YOLO format
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                bw /= w
                bh /= h

                # Class mapping
                # 1 = person → 0
                # others → vehicle → 1
                cls = 0 if ann["category_id"] == 1 else 1

                f.write(f"{cls} {xc} {yc} {bw} {bh}\n")

        count += 1

    print(f"✅ Labels created: {count} files in {label_dir}")


# =========================
# RUN BOTH
# =========================

convert(TRAIN_JSON, TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
convert(VAL_JSON, VAL_IMG_DIR, VAL_LABEL_DIR)

print("\n🔥 DONE: Labels generated successfully!")