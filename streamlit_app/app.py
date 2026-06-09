import streamlit as st
import pandas as pd
import os
import sys

# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Import project functions
try:
    from preprocess import clean_text_core
except:
    pass

try:
    from predict import run_single_inference
except:
    pass

try:
    from credibility_score import (
        compute_credibility_index,
        assign_credibility_label
    )
except:
    pass

st.set_page_config(
    page_title="AI-Powered Deceptive Review Detection System",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI-Powered Deceptive Review Detection System")
st.markdown(
    "Detect deceptive reviews using NLP, Machine Learning, Credibility Scoring and Similarity Analysis."
)

# ------------------------
# TABS
# ------------------------

tab1, tab2 = st.tabs(
    [
        "🎯 Ad-Hoc Inference Testing",
        "📊 Enterprise Batch Dashboard"
    ]
)

# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.subheader("Review Authenticity Verification")

    review_text = st.text_area(
        "Enter a Review",
        height=150,
        placeholder="Type or paste a review..."
    )

    if st.button("Verify Review Authenticity"):

        if review_text.strip() == "":
            st.warning("Please enter a review.")
        else:

            try:

                result = run_single_inference(review_text)

                prediction = result.get(
                    "prediction",
                    "Unknown"
                )

                confidence = result.get(
                    "confidence",
                    0
                )

           except Exception as e:

                st.error(f"Prediction Error: {e}")
            
                prediction = "Unavailable"
                confidence = 0

            try:

                credibility_score = compute_credibility_index(
                    review_text
                )

                credibility_label = assign_credibility_label(
                    credibility_score
                )

            except Exception as e:

                    st.error(f"Credibility Error: {e}")
                
                    credibility_score = 0
                    credibility_label = "Unknown"

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Prediction",
                    prediction
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with col3:
                st.metric(
                    "Credibility",
                    f"{credibility_score:.2f}"
                )

            with col4:
                st.metric(
                    "Credibility Label",
                    credibility_label
                )

            st.success("Review analyzed successfully.")

# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.subheader("Enterprise Analytics Dashboard")

    csv_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "outputs",
        "results.csv"
    )

    if os.path.exists(csv_path):

        df = pd.read_csv(csv_path)

        st.markdown("### KPI Overview")

        total_reviews = len(df)

        deceptive_reviews = 0
        genuine_reviews = 0

        if "prediction" in df.columns:

            deceptive_reviews = len(
                df[
                    df["prediction"]
                    .astype(str)
                    .str.contains(
                        "deceptive",
                        case=False,
                        na=False
                    )
                ]
            )

            genuine_reviews = len(
                df[
                    df["prediction"]
                    .astype(str)
                    .str.contains(
                        "genuine",
                        case=False,
                        na=False
                    )
                ]
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Reviews",
                total_reviews
            )

        with col2:
            st.metric(
                "Genuine Reviews",
                genuine_reviews
            )

        with col3:
            st.metric(
                "Deceptive Reviews",
                deceptive_reviews
            )

        st.markdown("---")

        if "prediction" in df.columns:

            st.subheader(
                "Prediction Distribution"
            )

            st.bar_chart(
                df["prediction"].value_counts()
            )

        if "credibility_score" in df.columns:

            st.subheader(
                "Credibility Score Distribution"
            )

            st.line_chart(
                df["credibility_score"]
            )

        st.subheader("Dataset Preview")

        st.dataframe(
            df,
            use_container_width=True
        )

        csv_download = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Results CSV",
            csv_download,
            "results.csv",
            "text/csv"
        )

    else:

        st.error(
            "outputs/results.csv not found.\nRun export_results.py first."
        )
