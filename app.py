import streamlit as st
import pandas as pd
from utils import (
    preprocess_dataframe,
    create_tfidf_matrix,
    compute_self_similarity,
    compute_metrics,
    match_user_input,
    match_icd_description,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_possible_diagnoses
)

st.set_page_config(page_title="CuramindAI", layout="wide")
st.title("🧠 CuramindAI: Medical Code Matcher Based On Medical Diagnosis")

tab1, tab2, tab3 = st.tabs(["📒Notebooks","📘 CCSR Matcher", "📙 ICD Matcher"])

# # ========== NOTEBOOK TAB ==========
with tab1:
    st.header("📝Notebook")
    tabC, tabI = st.tabs(["💉 CCSR", "💊 ICD"])
with tabC:
    uploaded_file = st.file_uploader("📂 Upload Cleaned Diagnosis CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File loaded successfully")
        if 'CCSR Diagnosis Description' not in df.columns or 'CCSR Diagnosis Code' not in df.columns:
            st.error("❌ CSV must contain 'CCSR Diagnosis Description' and 'CCSR Diagnosis Code' columns.")
            st.stop()

        df = preprocess_dataframe(df, 'CCSR Diagnosis Description')
        st.info("🔄 Vectorizing diagnosis descriptions...")
        vectorizer, tfidf_matrix = create_tfidf_matrix(df['CCSR Diagnosis Description'])

        df = compute_self_similarity(df, 'CCSR Diagnosis Description', 'CCSR Diagnosis Code', tfidf_matrix)

        if 'CCSR Diagnosis Code' in df.columns and 'Matched Code' in df.columns:
            y_true = df['CCSR Diagnosis Code'].astype(str)
            y_pred = df['Matched Code'].astype(str)

            accuracy, precision, recall, f1 = compute_metrics(y_true, y_pred)

            st.subheader("📊 Evaluation Metrics")
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("Precision", f"{precision:.4f}")
            st.metric("Recall", f"{recall:.4f}")
            st.metric("F1 Score", f"{f1:.4f}")
        else:
            st.warning("⚠️ Required columns not found for evaluation.")

        st.subheader("📝 Full Matched Results")
        st.dataframe(df[['CCSR Diagnosis Description', 'CCSR Diagnosis Code', 'Matched Description', 'Matched Code', 'Similarity Score']].head(100))

        st.download_button("💾 Download Final Matched CSV", df.to_csv(index=False), "final_matched_results.csv", "text/csv")

# ========== CCSR TAB ==========
with tab2:
    st.header("📘 CCSR Diagnosis Matcher")
    st.subheader("🔍 Try Your Own Diagnosis (CCSR)")
    user_input = st.text_area("Enter diagnosis description for CCSR")
    if st.button("Match CCSR Diagnosis"):
            if user_input.strip() == "":
                st.warning("Please enter a diagnosis description to find matches.")
            else:
                with st.spinner("🔄 Processing diagnosis and finding best match..."):
                    df_ccsr = pd.read_csv("./data/ccsr/cleaned_medical_dataset.csv")
                    df_ccsr = preprocess_dataframe(df_ccsr, 'CCSR Diagnosis Description')
                    vectorizer, tfidf_matrix = create_tfidf_matrix(df_ccsr['CCSR Diagnosis Description'])
                    df_ccsr = compute_self_similarity(df_ccsr, 'CCSR Diagnosis Description', 'CCSR Diagnosis Code', tfidf_matrix)
                    top_match, top_code, top_cost, top_score = match_user_input(
                        user_input, vectorizer, tfidf_matrix, df_ccsr,
                        'CCSR Diagnosis Description', 'CCSR Diagnosis Code', 'Total Costs',
                        threshold=0.5, flag="CCSR"
                    )
                    if top_match is None:
                        st.warning(f"⚠️ No match found. Highest similarity: {top_score:.4f}")
                    else:
                        st.markdown(f"**Top Match:** {top_match}")
                        st.markdown(f"**CCSR Code:** `{top_code}`")
                        st.markdown(f"**Similarity Score:** `{top_score:.4f}`")
                        st.markdown(f"**Estimated Cost:** ${top_cost:,.2f}")
    st.subheader("📃 Try by Uploading the Diagnosis")
    uploaded_file = st.file_uploader("Upload a document with embedded diagnoses", type=["pdf", "docx"], key="doc_match")
    text = ""
    if uploaded_file:
        with st.spinner("🔄 Extracting text, processing diagnosis, and generating matches..."):
            if uploaded_file.name.endswith(".pdf"):
                text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.endswith(".docx"):
                text = extract_text_from_docx(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a PDF or DOCX file.")
                st.stop()
            print("Extracted text:", text, type(text))
            diagnosis = extract_possible_diagnoses(text)
            print("Extracted diagnoses:", diagnosis, len(diagnosis))
            results = []
            finalResults = []
            df_ccsr = pd.read_csv("./data/ccsr/cleaned_medical_dataset.csv")
            df_ccsr = preprocess_dataframe(df_ccsr, 'CCSR Diagnosis Description')
            vectorizer, tfidf_matrix = create_tfidf_matrix(df_ccsr['CCSR Diagnosis Description'])
            df_ccsr = compute_self_similarity(df_ccsr, 'CCSR Diagnosis Description', 'CCSR Diagnosis Code', tfidf_matrix)
            for diagnosis in diagnosis:
                top_match, top_code, top_cost, top_score = match_user_input(
                    diagnosis, vectorizer, tfidf_matrix, df_ccsr,
                    'CCSR Diagnosis Description', 'CCSR Diagnosis Code', 'Total Costs',
                    threshold=0.5, flag="CCSR"
                )
                results.append({
                    "Diagnosis": diagnosis,
                    "Top Match": top_match,
                    "CCSR Code": top_code,
                    "Similarity Score": top_score,
                    "Estimated Cost": top_cost
                })
            if results:
                results_df = pd.DataFrame(results)
                # st.subheader("📋 Matched Results from Uploaded Document")
                # st.dataframe(results_df)
                st.download_button("💾 Download Matched Results", results_df.to_csv(index=False), "matched_results.csv", "text/csv")
                for res in results:
                    finalResults.append({"Similarity Score": float(f"{res['Similarity Score']:.4f}")})
                    if res.get("Similarity Score") > 0.5 and res.get("Top Match") is not None:
                        st.markdown(f"**Diagnosis:** {res['Diagnosis']}")
                        st.markdown(f"**Top Match:** {res['Top Match']}")
                        st.markdown(f"**CCSR Code:** `{res['CCSR Code']}`")
                        st.markdown(f"**Similarity Score:** `{res['Similarity Score']:.4f}`")
                        st.markdown(f"**Estimated Cost:** ${res['Estimated Cost']:,.2f}")
                        st.markdown("---")


# ========== ICD TAB ==========

with tab3:
    st.header("📙 ICD-10 Code Matcher")
    icd_input = st.text_area("Enter diagnosis description for ICD")
    if st.button("Match ICD Code"):
        if icd_input.strip() == "":
            st.warning("Please enter a diagnosis description to find matches.")
        elif icd_input:
            with st.spinner("🔄 Processing diagnosis and finding best match..."):
                top_match, top_code, _, top_score = match_icd_description(icd_input)
                if top_match is None:
                    st.warning(f"⚠️ No good match found. Highest similarity score: {top_score:.4f}")
                else:
                    st.markdown(f"**Top Match:** {top_match}")
                    st.markdown(f"**ICD Code:** `{top_code}`")
                    st.markdown(f"**Similarity Score:** `{top_score:.4f}`")
        else:
            st.warning("Please enter a diagnosis description.")
    st.subheader("📃 Try by Uploading the Diagnosis")
    uploaded_file = st.file_uploader("Upload a document with embedded diagnoses", type=["pdf", "docx"], key="doc_match_icd")