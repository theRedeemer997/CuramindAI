# app.py

import streamlit as st
import pandas as pd
from utils import (
    preprocess_dataframe,
    create_tfidf_matrix,
    compute_self_similarity,
    compute_metrics,
    match_user_input
)

st.set_page_config(page_title="Medical Code Matcher", layout="wide")
st.title("🧠 CuramindAI: Medical Code Matcher Based On Medical Diagnosis")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload Cleaned Diagnosis CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File loaded successfully")

    # Check required columns
    if 'CCSR Diagnosis Description' not in df.columns or 'CCSR Diagnosis Code' not in df.columns:
        st.error("❌ CSV must contain 'CCSR Diagnosis Description' and 'CCSR Diagnosis Code' columns.")
        st.stop()

    # Clean and preprocess
    df = preprocess_dataframe(df, 'CCSR Diagnosis Description')

    # TF-IDF Vectorization
    st.info("🔄 Vectorizing diagnosis descriptions...")
    vectorizer, tfidf_matrix = create_tfidf_matrix(df['CCSR Diagnosis Description'])

    # Compute similarity and matched results
    df = compute_self_similarity(df, 'CCSR Diagnosis Description', 'CCSR Diagnosis Code', tfidf_matrix)

    # Evaluation metrics
    if 'CCSR Diagnosis Code' in df.columns and 'Matched Code' in df.columns:
        y_true = df['CCSR Diagnosis Code'].astype(str)
        y_pred = df['Matched Code'].astype(str)

        accuracy, precision, recall, f1 = compute_metrics(y_true, y_pred)

        st.subheader("📊 Diagnosis Code-Level Evaluation Metrics")
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")
        st.metric("F1 Score", f"{f1:.4f}")
    else:
        st.warning("⚠️ Columns 'CCSR Diagnosis Code' or 'Matched Code' not found in uploaded file.")

    # Display results
    st.subheader("📝 Full Matched Results")
    st.dataframe(df[['CCSR Diagnosis Description', 'CCSR Diagnosis Code', 'Matched Description', 'Matched Code', 'Similarity Score']].head(100))

    st.download_button("💾 Download Final Matched CSV", df.to_csv(index=False), "final_matched_results.csv", "text/csv")

    # User input matching
    st.subheader("🔍 Try Your Own Diagnosis")
    user_input = st.text_area("Enter diagnosis description (e.g., 'acute myocardial infarction')")
    SIMILARITY_SCORE_THRESHOLD = 0.5

    if st.button("Match Diagnosis"):
        if user_input.strip():
            top_match, top_code, top_cost, top_score = match_user_input(
                user_input, vectorizer, tfidf_matrix, df,
                'CCSR Diagnosis Description', 'CCSR Diagnosis Code', 'Total Costs',
                threshold=SIMILARITY_SCORE_THRESHOLD
            )

            if top_match is None:
                st.warning(f"⚠️ No good match found. Highest similarity score: {top_score:.4f}")
            else:
                st.markdown(f"**Top Match:** {top_match}")
                st.markdown(f"**CCSR Code:** `{top_code}`")
                st.markdown(f"**Similarity Score:** `{top_score:.4f}`")
                st.markdown(f"**Estimated Cost:** ${top_cost:,.2f}")
        else:
            st.warning("Please enter a diagnosis description to match.")
