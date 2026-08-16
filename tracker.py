import cv2
import numpy as np
import time
from ultralytics import YOLO
from supabase import create_client, Client

# --- SUPABASE SETUP ---
SUPABASE_URL = "YOUR_SUPABASE_PROJECT_URL_HERE"
SUPABASE_KEY = "YOUR_SUPABASE_ANON_KEY_HERE"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def format_time(seconds):
    if seconds < 60: return f"{seconds}s"
    elif seconds < 3600: return f"{seconds // 60}m {seconds % 60}s"
    else: return f"{seconds // 3600}h {(seconds % 3600) // 60}m"

model = YOLO("yolo11n.pt")

dwell_timers = {}
last_seen = {}
GRACE_PERIOD = 300.0  # 5 minutes memory

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
zone_polygon = np.array([[0, 0], [frame_width, 0], [frame_width, frame_height], [0, frame_height]], np.int32)

last_process_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    current_time = time.time()

    if current_time - last_process_time >= 1.0:
        last_process_time = current_time

        # Using BoT-SORT for Re-ID and Mac GPU for efficiency
        results = model.track(frame, persist=True, tracker="botsort.yaml", classes=[0], device="mps", imgsz=320)

        cv2.polylines(frame, [zone_polygon], True, (0, 255, 255), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                foot_x, foot_y = int((x1 + x2) / 2), int(y2)

                if cv2.pointPolygonTest(zone_polygon, (foot_x, foot_y), False) >= 0:
                    if track_id not in dwell_timers:
                        dwell_timers[track_id] = current_time

                    last_seen[track_id] = current_time
                    time_spent_raw = int(current_time - dwell_timers[track_id])

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"ID: {track_id} | {format_time(time_spent_raw)}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 5-Minute Grace Period Check
        expired_ids = [t_id for t_id in list(dwell_timers.keys()) if current_time - last_seen.get(t_id, current_time) > GRACE_PERIOD]

        for t_id in expired_ids:
            final_time_raw = int(last_seen[t_id] - dwell_timers[t_id])
            if final_time_raw > 2:
                # PUSH TO CLOUD DATABASE (Robust network handling)
                try:
                    supabase.table("cafe_data").insert({
                        "person_id": f"Person_{t_id}",
                        "formatted_time": format_time(final_time_raw),
                        "raw_seconds": final_time_raw
                    }).execute()
                    print(f"✅ Uploaded Person_{t_id} to cloud.")
                except Exception as e:
                    print(f"❌ Network error. Failed to upload Person_{t_id}: {e}")

            del dwell_timers[t_id]
            del last_seen[t_id]

        cv2.imshow("Cafe Tracker (Cloud Edition)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
