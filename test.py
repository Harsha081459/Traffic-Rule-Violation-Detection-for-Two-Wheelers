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



# from solution_final_strong import TrafficViolationDetector
# 
# import json
# import time
# import cv2
# 
# IMAGE_PATH = r"D:\draft2_cv\image copy 9.png"
# 
# # Model initialization is done once by the evaluator, so this is not included in inference time.
# detector = TrafficViolationDetector(model_dir="./models")
# 
# # For speed testing, use predict(). For visual debugging, use predict_debug() if available.
# t_start = time.perf_counter()
# if hasattr(detector, "predict_debug"):
#     result = detector.predict_debug(IMAGE_PATH)
# else:
#     result = detector.predict(IMAGE_PATH)
# inference_time = round(time.perf_counter() - t_start, 3)
# 
