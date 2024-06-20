# streamlit_app.py
import streamlit as st
import pandas as pd
import datetime
import altair as alt
from send_email import send_email, async_send_email

CLASS_NAMES = ["belly pain", "burping", "discomfort", "hungry", "tired"]

@st.cache_data(ttl=60)
def load_data():
    df = pd.read_csv('./data/detections.csv', names=["Timestamp", "Classification"])
    return df

# Read the application status from the text file and display it in the top right corner
with open('app_current_status.txt', 'r') as file:
    app_status = file.read()
st.markdown(f"<h6 style='text-align: right; color: white;'>Application Status: {app_status}</h6>", unsafe_allow_html=True)

# Add a button for refreshing data
if st.button('Refresh Data'):
    df = load_data()
    st.experimental_rerun()

# Load data
df = load_data()

# Ensure that there is data
if df.shape[0] == 0:
    st.markdown("<h1 style='text-align: center; color: white;'>No Data Currently Available</h1>", unsafe_allow_html=True)
    st.stop()

latest_classification = df["Classification"].iloc[-1]
latest_timestamp = df["Timestamp"].iloc[-1]
st.markdown(f"<h1 style='text-align: center; color: white;'>Sophia's Latest Cry Prediction: {latest_classification}</h1>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align: center; color: white;'>Time: {latest_timestamp}</h1>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("All Detections:", unsafe_allow_html=True)
st.dataframe(df.sort_values(by="Timestamp", ascending=False), height=600, width=800)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

cry_counts = df["Classification"].value_counts().reset_index()
cry_counts.columns = ["Cry Type", "Count"]

chart = alt.Chart(cry_counts).mark_bar().encode(
    x=alt.X("Cry Type", axis=alt.Axis(labelAngle=0), sort=CLASS_NAMES),  # Horizontal labels
    y="Count"
)

st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
st.altair_chart(chart, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

