from solution_final_strong import TrafficViolationDetector

import json
import time
import cv2

IMAGE_PATH = r"D:\draft2_cv\image copy 9.png"

# Model initialization is done once by the evaluator, so this is not included in inference time.
detector = TrafficViolationDetector(model_dir="./models")

# For speed testing, use predict(). For visual debugging, use predict_debug() if available.
t_start = time.perf_counter()
if hasattr(detector, "predict_debug"):
    result = detector.predict_debug(IMAGE_PATH)
else:
    result = detector.predict(IMAGE_PATH)
inference_time = round(time.perf_counter() - t_start, 3)

output = {"violations": result.get("violations", [])}
print(json.dumps(output, indent=2))
print(f"\nInference time: {inference_time:.3f} s")

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

debug_items = result.get("debug", [])

for idx, item in enumerate(debug_items):
    is_violation = item.get("is_violation", False)

    x1, y1, x2, y2 = item["bike_bbox"]
    bike_color = (0, 0, 255) if is_violation else (0, 255, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), bike_color, 3)

    label = (
        f"Bike {idx + 1} | riders={item.get('num_riders', 0)} | "
        f"no_helmet={item.get('helmet_violations', 0)} | "
        f"plate={item.get('license_plate', '') or 'N/A'}"
    )
    cv2.putText(img, label, (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, bike_color, 2, cv2.LINE_AA)

    for rb in item.get("rider_bboxes", []):
        rx1, ry1, rx2, ry2 = rb
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
        cv2.putText(img, "rider", (rx1, max(20, ry1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    for hb in item.get("helmet_bboxes", []):
        hx1, hy1, hx2, hy2 = hb
        cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
        cv2.putText(img, "helmet", (hx1, max(20, hy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    for nhb in item.get("no_helmet_bboxes", []):
        nx1, ny1, nx2, ny2 = nhb
        cv2.rectangle(img, (nx1, ny1), (nx2, ny2), (0, 0, 255), 2)
        cv2.putText(img, "no_helmet", (nx1, max(20, ny1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

time_label = f"Inference: {inference_time:.3f}s"
(tw, th), _ = cv2.getTextSize(time_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
cv2.rectangle(img, (8, 8), (18 + tw, 18 + th), (0, 0, 0), -1)
cv2.putText(img, time_label, (12, 12 + th),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

max_width, max_height = 1200, 800
h, w = img.shape[:2]
scale = min(max_width / w, max_height / h, 1.0)
display_img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA) if scale < 1.0 else img

cv2.imshow("Traffic Violation Detection", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
