import streamlit as st
import pandas as pd
import numpy as np
import time
from ultralytics import YOLO
from supabase import create_client, Client
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

# Page Configuration
st.set_page_config(layout="wide", page_title="Live Cafe Tracker & Dashboard")

st.title("☕ Live Cafe Dwell Tracker & Dashboard")

# Securely grab Supabase credentials from Streamlit Secrets
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load YOLO Model (cached so it only loads once)
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# Time formatting helper
def format_time(seconds):
    if seconds < 60: return f"{seconds}s"
    elif seconds < 3600: return f"{seconds // 60}m {seconds % 60}s"
    else: return f"{seconds // 3600}h {(seconds % 3600) // 60}m"

# WebRTC Configuration for live video streaming
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Layout: Two columns (Left: Live Camera & AI, Right: Database Analytics)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔴 Live Camera Feed & AI Tracking")

    # Session state variables to handle tracking timers across video frames
    if "dwell_timers" not in st.session_state:
        st.session_state.dwell_timers = {}
    if "last_seen" not in st.session_state:
        st.session_state.last_seen = {}

    GRACE_PERIOD = 300.0  # 5 minutes memory

    # Video processing callback function for Streamlit-WebRTC
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()

        # Get dimensions
        height, width, _ = img.shape
        zone_polygon = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.int32)

        # Run YOLO tracking with BoT-SORT Re-ID
        results = model.track(img, persist=True, tracker="botsort.yaml", classes=[0], imgsz=320)

        # Draw full screen zone boundary
        import cv2
        cv2.polylines(img, [zone_polygon], True, (0, 255, 255), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                foot_x, foot_y = int((x1 + x2) / 2), int(y2)

                if cv2.pointPolygonTest(zone_polygon, (foot_x, foot_y), False) >= 0:
                    if track_id not in st.session_state.dwell_timers:
                        st.session_state.dwell_timers[track_id] = current_time

                    st.session_state.last_seen[track_id] = current_time
                    time_spent_raw = int(current_time - st.session_state.dwell_timers[track_id])

                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"ID: {track_id} | {format_time(time_spent_raw)}"
                    cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Grace period checkout logic
        expired_ids = [t_id for t_id in list(st.session_state.dwell_timers.keys())
                       if current_time - st.session_state.last_seen.get(t_id, current_time) > GRACE_PERIOD]

        for t_id in expired_ids:
            final_time_raw = int(st.session_state.last_seen[t_id] - st.session_state.dwell_timers[t_id])
            if final_time_raw > 2:
                try:
                    supabase.table("cafe_data").insert({
                        "person_id": f"Person_{t_id}",
                        "formatted_time": format_time(final_time_raw),
                        "raw_seconds": final_time_raw
                    }).execute()
                except Exception:
                    pass

            del st.session_state.dwell_timers[t_id]
            del st.session_state.last_seen[t_id]

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Launch the WebRTC video streaming component on the website
    webrtc_streamer(
        key="cafe-tracker",
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    st.subheader("📊 Live Database Analytics")

    # Pull data from Supabase to show real-time stats
    try:
        response = supabase.table("cafe_data").select("*").execute()
        data = response.data

        if not data:
            st.info("No completed sessions logged in the database yet.")
        else:
            df = pd.DataFrame(data)

            avg_minutes = (df["raw_seconds"].mean()) / 60
            st.metric(label="Average Dwell Time", value=f"{avg_minutes:.1f} minutes")

            st.subheader("Customer History Chart")
            st.bar_chart(df, x="person_id", y="raw_seconds")

            st.dataframe(df[["person_id", "formatted_time", "raw_seconds", "created_at"]], use_container_width=True)

    except Exception as e:
        st.error(f"Database connection error: {e}")

# Refresh the dashboard view every 15 seconds
time.sleep(15)
st.rerun()
