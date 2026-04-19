from ultralytics import YOLO
import cv2

# =========================
# Load trained model
# =========================
model = YOLO("runs/detect/train/weights/best.pt")

# =========================
# Load thermal image
# =========================
img = cv2.imread("D:/AI project/test.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("❌ Image not found")
    exit()

# =========================
# Improve contrast (IMPORTANT)
# =========================
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
img = clahe.apply(img)

# Convert to 3-channel
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# =========================
# Run detection
# =========================
results = model(img, conf=0.3)

# =========================
# Counting
# =========================
count_person = 0
count_vehicle = 0

for box in results[0].boxes:
    cls = int(box.cls[0])

    if cls == 0:
        count_person += 1
    elif cls == 1:
        count_vehicle += 1

print(f"👤 Humans: {count_person}")
print(f"🚗 Vehicles: {count_vehicle}")

# =========================
# Show output
# =========================
annotated = results[0].plot()

cv2.imshow("Thermal Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()