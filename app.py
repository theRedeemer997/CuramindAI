import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

st.set_page_config(page_title="Medical Code Matcher", layout="wide")
st.title("🧠 CuramindAI: Medical Code Matcher Based On Medical Diagnosis")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload Cleaned Diagnosis CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File loaded successfully")

    # Check required columns
    # Ensure the CSV contains necessary columns for matching, else show error
    if 'CCSR Diagnosis Description' not in df.columns or 'CCSR Diagnosis Code' not in df.columns:
        st.error("❌ CSV must contain 'CCSR Diagnosis Description' and 'CCSR Diagnosis Code' columns.")
        st.stop()

    # Remove duplicates and clean
    # Ensuring all descriptions are unique and properly formatted and are strings
    df = df.drop_duplicates(subset='CCSR Diagnosis Description')
    df['CCSR Diagnosis Description'] = df['CCSR Diagnosis Description'].astype(str)

    # TF-IDF Vectorization
    st.info("🔄 Vectorizing diagnosis descriptions...")
    # Using TF-IDF to convert text data into numerical format
    # so in the below lines, we create a TF-IDF vectorizer and fit it to the diagnosis descriptions 
    #  convert text data into a matrix of TF-IDF features and then compute the cosine similarity between these vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['CCSR Diagnosis Description'])

    # Compute similarity (self-match for evaluation)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Find best match for each row (excluding self-match if desired)
    # Here we find the index of the maximum similarity score for each diagnosis description
    # and then extract the corresponding matched description and code
    # This gives us the best match for each diagnosis based on the TF-IDF cosine similarity
    best_match_indices = similarity_matrix.argmax(axis=1)
    best_scores = similarity_matrix.max(axis=1)
    best_matches = [df.iloc[i]['CCSR Diagnosis Description'] for i in best_match_indices]
    match_codes = [df.iloc[i]['CCSR Diagnosis Code'] for i in best_match_indices]

    # Add results to DataFrame
    df['Matched Description'] = best_matches
    df['Matched Code'] = match_codes
    df['Similarity Score'] = best_scores

    # Evaluation metric (mean similarity)
    mean_similarity = df['Similarity Score'].mean()

    st.subheader("📊 Evaluation Metric")
    st.metric("🔎 Mean Similarity Score", f"{mean_similarity:.4f}")
    st.caption("Note: Similarity score is based on TF-IDF cosine similarity between diagnosis descriptions.")

    # Filter by similarity threshold
    threshold = st.slider("🔍 Filter by Similarity Score", 0.0, 1.0, 0.75)
    low_conf = df[df['Similarity Score'] < threshold]
    # st.subheader(f"⚠️ Low-Confidence Matches (< {threshold})")
    # st.dataframe(low_conf[['CCSR Diagnosis Description', 'Matched Description', 'Matched Code', 'Similarity Score']])

    # Display accuracy add other metrics
    if 'CCSR Diagnosis Code' in df.columns and 'Matched Code' in df.columns:
        y_true = df['CCSR Diagnosis Code'].astype(str)
        y_pred = df['Matched Code'].astype(str)


        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        # Display in Streamlit
        st.subheader("📊 Diagnosis Code-Level Evaluation Metrics")
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")
        st.metric("F1 Score", f"{f1:.4f}")
    else:
        st.warning("⚠️ Columns 'CCSR Diagnosis Code' or 'Matched Code' not found in uploaded file.")

    

    st.subheader("📝 Full Matched Results")
    st.dataframe(df[['CCSR Diagnosis Description', 'CCSR Diagnosis Code', 'Matched Description', 'Matched Code', 'Similarity Score']].head(100))

    st.download_button("💾 Download Final Matched CSV", df.to_csv(index=False), "final_matched_results.csv", "text/csv")

    # User input for live diagnosis matching
    st.subheader("🔍 Try Your Own Diagnosis")
    user_input = st.text_area("Enter diagnosis description (e.g., 'acute myocardial infarction')")

    # Now we try to simulate the real-time matching process
    # When the user inputs a diagnosis description, we vectorize it and compute its similarity with the existing TF-IDF matrix
    # This allows us to find the most similar diagnosis in the dataset 
    # and display the top match along with its code and estimated cost
    if st.button("Match Diagnosis"):
        if user_input.strip():
            query_vec = vectorizer.transform([user_input])
            sim_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
            top_idx = sim_scores.argmax()
            top_score = sim_scores[top_idx]
            top_match = df.iloc[top_idx]['CCSR Diagnosis Description']
            top_code = df.iloc[top_idx]['CCSR Diagnosis Code']
            top_cost = df.iloc[top_idx]['Total Costs']

            st.markdown(f"**Top Match:** {top_match}")
            st.markdown(f"**CCSR Code:** `{top_code}`")
            st.markdown(f"**Similarity Score:** `{top_score:.4f}`")
            st.markdown(f"**Estimated Cost:** ${top_cost:,.2f}")
        else:
            st.warning("Please enter a diagnosis description to match.")
