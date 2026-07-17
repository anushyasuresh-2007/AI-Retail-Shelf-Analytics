import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
from PIL import Image


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Retail Shelf Analytics",
    page_icon="🛒",
    layout="wide"
)


# ==========================================
# CUSTOM STYLE
# ==========================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    h1 {
        font-size: 2.5rem;
    }

    [data-testid="stMetric"] {
        border: 1px solid #d9d9d9;
        padding: 15px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🛒 Retail Analytics")

st.sidebar.markdown(
    """
    ### Navigation

    📊 Shelf Analytics

    🔥 Customer Heatmap

    🎥 Tracking Video

    💡 Business Insights
    """
)

st.sidebar.divider()

st.sidebar.success(
    "AI Analysis Active"
)

st.sidebar.info(
    "Powered by YOLO Computer Vision"
)


# ==========================================
# VIDEO UPLOAD
# ==========================================

st.sidebar.divider()

st.sidebar.subheader(
    "📹 Upload Surveillance Video"
)

uploaded_video = st.sidebar.file_uploader(
    "Choose a video file",
    type=["avi", "mp4", "mov"]
)

if uploaded_video is not None:

    st.sidebar.success(
        "Video uploaded successfully!"
    )

    st.video(
        uploaded_video
    )


# ==========================================
# LOAD ANALYTICS JSON
# ==========================================

try:

    with open(
        "analytics.json",
        "r"
    ) as file:

        data = json.load(file)

except FileNotFoundError:

    st.error(
        "analytics.json not found. "
        "Run shelf_analytics.py first."
    )

    st.stop()


# ==========================================
# TITLE
# ==========================================

st.title(
    "🛒 AI-Powered Retail Shelf Engagement Analytics"
)

st.markdown(
    "Computer Vision-Based Customer Behavior and Shelf Engagement Analysis"
)

st.divider()


# ==========================================
# CREATE DATAFRAME
# ==========================================

shelf_rows = []


for shelf_name, values in data[
    "shelves"
].items():

    shelf_rows.append(

        {
            "Shelf": shelf_name,

            "Visitors": values[
                "visitors"
            ],

            "Total Dwell Time (sec)": values[
                "total_dwell_time"
            ],

            "Average Dwell Time (sec)": values[
                "average_dwell_time"
            ]
        }
    )


df = pd.DataFrame(
    shelf_rows
)


# ==========================================
# KEY METRICS
# ==========================================

total_visitors = data[
    "total_unique_visitors"
]

most_visited = data[
    "most_visited_shelf"
]

least_visited = data[
    "least_visited_shelf"
]

highest_engagement = data[
    "highest_engagement_shelf"
]

total_dwell = df[
    "Total Dwell Time (sec)"
].sum()


peak_dwell = df[
    "Average Dwell Time (sec)"
].max()


# ==========================================
# METRIC CARDS
# ==========================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "👥 Total Unique Visitors",
        total_visitors
    )


with col2:

    st.metric(
        "🏆 Most Visited Shelf",
        most_visited
    )


with col3:

    st.metric(
        "🔥 Highest Engagement",
        highest_engagement
    )


with col4:

    st.metric(
        "⏱ Total Dwell Time",
        f"{total_dwell:.2f} sec"
    )


st.divider()


# ==========================================
# SHELF PERFORMANCE
# ==========================================

st.subheader(
    "📊 Shelf Performance Analytics"
)


st.dataframe(
    df,
    width="stretch",
    hide_index=True
)


# ==========================================
# DOWNLOAD REPORT
# ==========================================

st.subheader(
    "📥 Download Analytics Report"
)


csv_data = df.to_csv(
    index=False
)


st.download_button(
    label="Download Shelf Analytics CSV",
    data=csv_data,
    file_name="shelf_analytics_report.csv",
    mime="text/csv",
    width="stretch"
)


# ==========================================
# CHARTS
# ==========================================

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "👥 Visitors per Shelf"
    )


    fig1, ax1 = plt.subplots()


    ax1.bar(
        df["Shelf"],
        df["Visitors"]
    )


    ax1.set_xlabel(
        "Shelf"
    )


    ax1.set_ylabel(
        "Number of Visitors"
    )


    ax1.set_title(
        "Shelf Visitor Count"
    )


    st.pyplot(
        fig1
    )


with col2:

    st.subheader(
        "⏱ Average Dwell Time"
    )


    fig2, ax2 = plt.subplots()


    ax2.bar(
        df["Shelf"],
        df[
            "Average Dwell Time (sec)"
        ]
    )


    ax2.set_xlabel(
        "Shelf"
    )


    ax2.set_ylabel(
        "Average Time (seconds)"
    )


    ax2.set_title(
        "Customer Engagement by Shelf"
    )


    st.pyplot(
        fig2
    )


# ==========================================
# SHELF ENGAGEMENT RANKING
# ==========================================

st.divider()


st.subheader(
    "🏆 Shelf Engagement Ranking"
)


ranking = df.sort_values(
    by="Average Dwell Time (sec)",
    ascending=False
).reset_index(
    drop=True
)


ranking.index = ranking.index + 1


st.dataframe(
    ranking[
        [
            "Shelf",
            "Visitors",
            "Average Dwell Time (sec)"
        ]
    ],
    width="stretch"
)


# ==========================================
# CUSTOMER TRAFFIC DISTRIBUTION
# ==========================================

st.divider()


st.subheader(
    "🚶 Customer Traffic Distribution"
)


fig3, ax3 = plt.subplots()


ax3.pie(
    df["Visitors"],
    labels=df["Shelf"],
    autopct="%1.1f%%"
)


ax3.set_title(
    "Distribution of Customers Across Shelves"
)


st.pyplot(
    fig3
)


# ==========================================
# HEATMAP
# ==========================================

st.divider()


st.subheader(
    "🔥 Customer Movement Heatmap"
)


try:

    image = Image.open(
        "customer_heatmap.jpg"
    )


    st.image(
        image,
        width="stretch"
    )


except FileNotFoundError:

    st.warning(
        "Heatmap image not found. "
        "Run shelf_analytics.py first."
    )


# ==========================================
# ANNOTATED VIDEO
# ==========================================

st.divider()


st.subheader(
    "🎥 Annotated Customer Tracking Video"
)


video_path = (
    "runs/detect/track/"
    "Copy of Alif Store-Viewing-10.avi"
)


try:

    st.video(
        video_path
    )


except Exception:

    st.warning(
        "Annotated video not found."
    )


# ==========================================
# PEAK ENGAGEMENT
# ==========================================

st.divider()


st.subheader(
    "⏰ Peak Engagement Analysis"
)


st.success(
    f"Peak engagement was recorded at "
    f"{highest_engagement} "
    f"with an average dwell time of "
    f"{peak_dwell:.2f} seconds."
)


st.info(
    "The current peak engagement analysis is shelf-based. "
    "Time-of-day analysis can be added when timestamped "
    "surveillance data is available."
)


# ==========================================
# BUSINESS INSIGHTS
# ==========================================

st.divider()


st.subheader(
    "💡 Business Insights"
)


st.info(
    f"""
• {most_visited} attracted the highest number of visitors.

• {highest_engagement} had the highest average customer engagement.

• {least_visited} was the least visited shelf.

• The most visited shelf is not necessarily the most engaging shelf.

• Retailers can use these insights to optimize product placement.

• High dwell time indicates stronger customer attention.

• Customer traffic distribution helps identify high-traffic store zones.
"""
)


# ==========================================
# FOOTER
# ==========================================

st.divider()


st.caption(
    "AI-Powered Retail Shelf Engagement Analytics | "
    "Computer Vision Project"
)