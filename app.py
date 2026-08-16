import streamlit as st
import pandas as pd
import time
from supabase import create_client, Client

st.set_page_config(layout="wide", page_title="Live Cafe Data")

# Securely grab credentials from Streamlit Secrets (configured in Step 4)
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("☕ Live Cafe Dwell Times")

try:
    # Fetch data from Supabase
    response = supabase.table("cafe_data").select("*").execute()
    data = response.data

    if not data:
        st.info("No data yet. Waiting for customers...")
    else:
        df = pd.DataFrame(data)

        # Calculate Average Time
        avg_seconds = df["raw_seconds"].mean()
        avg_minutes = avg_seconds / 60
        st.metric(label="Average Dwell Time Today", value=f"{avg_minutes:.1f} minutes")

        # Draw the chart
        st.subheader("Customer History")
        st.bar_chart(df, x="person_id", y="raw_seconds")

        # Display clean table (hiding the internal Supabase IDs)
        st.dataframe(df[["person_id", "formatted_time", "raw_seconds", "created_at"]], use_container_width=True)

except Exception as e:
    st.error(f"Failed to connect to database: {e}")

# Auto-refresh the dashboard every 10 seconds to get live cloud updates
time.sleep(10)
st.rerun()
