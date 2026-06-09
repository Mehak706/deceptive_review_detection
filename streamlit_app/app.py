import streamlit as st
import pandas as pd
import os
import sys

# ==================================================
# PATH SETUP
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "..", "src")

sys.path.append(SRC_DIR)

# ==================================================
# IMPORTS
# ==================================================

try:
    from predict import run_single_inference
    predict_loaded = True
except Exception as e:
    predict_loaded = False
    predict_error = str(e)

try:
    from credibility_score import assign_credibility_label
except Exception as e:
    st.error(f"Credibility Import Error: {e}")

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI-Powered Deceptive Review Detection System",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI-Powered Deceptive Review Detection System")

st.markdown(
    """
    Detect deceptive reviews using NLP,
    Machine Learning and Credibility Analysis.
    """
)

# ==================================================
# IMPORT STATUS
# ==================================================

if predict_loaded:
    st.success("✅ Prediction Model Loaded Successfully")
else:
    st.error(f"❌ Predict Import Error: {predict_error}")

# ==================================================
# TABS
# ==================================================

tab1, tab2 = st.tabs(
    [
        "🎯 Review Verification",
        "📊 Enterprise Dashboard"
    ]
)

# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.subheader("Review Authenticity Verification")

    review_text = st.text_area(
        "Enter Review",
        height=180,
        placeholder="Paste review text here..."
    )

    if st.button("Verify Review Authenticity"):

        if review_text.strip() == "":
            st.warning("Please enter a review.")

        else:

            if not predict_loaded:
                st.error(
                    f"Prediction module not loaded.\n\n{predict_error}"
                )

            else:

                try:

                    result = run_single_inference(
                        review_text
                    )

                    prediction = result.get(
                        "prediction",
                        "Unknown"
                    )

                    confidence = (
                        result.get(
                            "confidence",
                            0
                        ) * 100
                    )

                    credibility_score = confidence

                    credibility_label = (
                        assign_credibility_label(
                            credibility_score
                        )
                    )

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
                            "Credibility Score",
                            f"{credibility_score:.2f}"
                        )

                    with col4:
                        st.metric(
                            "Credibility Label",
                            credibility_label
                        )

                    st.success(
                        "Review analyzed successfully."
                    )

                    with st.expander(
                        "View Raw Model Output"
                    ):
                        st.json(result)

                except Exception as e:

                    st.error(
                        f"Prediction Error: {e}"
                    )

# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.subheader("Enterprise Analytics Dashboard")

    csv_path = os.path.join(
        BASE_DIR,
        "..",
        "outputs",
        "results.csv"
    )

    if os.path.exists(csv_path):

        try:

            df = pd.read_csv(csv_path)

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

            st.subheader(
                "Dataset Preview"
            )

            st.dataframe(
                df,
                width="stretch"
            )

            csv_download = (
                df.to_csv(
                    index=False
                )
                .encode("utf-8")
            )

            st.download_button(
                label="Download Results CSV",
                data=csv_download,
                file_name="results.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                f"Dashboard Error: {e}"
            )

    else:

        st.warning(
            "outputs/results.csv not found."
        )
