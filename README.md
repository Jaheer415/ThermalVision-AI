# 🔥 ThermalVision-AI

Real-time Human & Vehicle Detection using Thermal Imaging and YOLOv8

---

## 📌 Overview

ThermalVision-AI is a deep learning-based object detection system designed to detect humans and vehicles from thermal images.
The model is fine-tuned on the FLIR thermal dataset using YOLOv8 and optimized for real-time inference.

This project demonstrates practical AI deployment using GPU acceleration and real-world dataset handling.

---

## 🚀 Features

* 🔥 Thermal image-based detection (works in low-light / night)
* 👤 Human detection
* 🚗 Vehicle detection
* ⚡ Real-time inference (GPU accelerated)
* 🧠 Fine-tuned YOLOv8 model
* 📊 Custom dataset training pipeline

---

## 🧠 Model Details

* Model: YOLOv8 (Ultralytics)
* Base Model: yolov8n.pt (fine-tuned)
* Input Size: 416x416
* Dataset: FLIR Thermal Dataset
* Classes:

  * 0 → Person
  * 1 → Vehicle

---

## 📂 Project Structure

```
AI project/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│
├── runs/
│   └── detect/
│       └── train/
│
├── thermal.yaml
├── convert_labels.py
└── detect.py
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ThermalVision-AI.git
cd ThermalVision-AI
```

### 2. Install dependencies

```bash
pip install ultralytics torch torchvision opencv-python
```

---

## 🏋️ Training

```bash
yolo detect train data=thermal.yaml model=yolov8n.pt epochs=40 imgsz=416 batch=4 device=0
```

---

## 🔍 Inference

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source="image.jpg" save=True
```

---

## 📊 Results

* Successfully detects humans in thermal images
* Works in night and low-visibility conditions
* Real-time inference on GPU (~20–30 ms per image)

---

## ⚠️ Limitations

* Struggles with:

  * Small or partially occluded objects
  * Unseen environments (e.g., forests)
* Requires more diverse data for generalization

---

## 🚀 Future Improvements

* Upgrade to YOLOv8s/m for higher accuracy
* Add more diverse thermal datasets
* Improve small object detection
* Deploy on edge devices (Jetson, drones)

---

## 🧠 Learnings

* Dataset quality > model size
* Fine-tuning significantly improves performance
* Real-world AI requires continuous iteration

---

## 🤝 Contribution

Feel free to fork and improve the project. Contributions are welcome!

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Udayagiri Jaheer
B.Tech Student | AI & Robotics Enthusiast

---

🔥 Built as part of hands-on learning in AI model training, deployment, and optimization.
