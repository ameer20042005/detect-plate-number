import os
import cv2
from fast_alpr import ALPR

# تحقق من وجود الفيديو
video_path = "video_2025-01-18_20-27-35.mp4"
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit()

# تهيئة ALPR
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)

# فتح الفيديو
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# الحصول على خصائص الفيديو
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# إعداد مسار الفيديو الناتج
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# قراءة الإطارات ومعالجتها
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # الكشف عن اللوحات
        alpr_results = alpr.predict(frame)

        # رسم النتائج
        annotated_frame = alpr.draw_predictions(frame)

        # حفظ الإطار في الفيديو الناتج
        out.write(annotated_frame)

        # عرض النتائج
        cv2.imshow("ALPR Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Error during processing: {e}")
finally:
    # تحرير الموارد
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print(f"تم حفظ الفيديو الناتج في: {output_path}")
