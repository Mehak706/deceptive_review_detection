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
