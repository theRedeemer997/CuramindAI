# Adding the necessary imports for utils.py
import fitz  
import docx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import spacy

nlp = spacy.load("en_core_web_sm")

# Cleaned and prepeared DataFrame
def preprocess_dataframe(df, description_col):
    df = df.drop_duplicates(subset=description_col)
    df[description_col] = df[description_col].astype(str)
    return df

# Create TF-IDF matrix
def create_tfidf_matrix(text_data):
    # initialize the vectorizer
    vectorizer = TfidfVectorizer()
    # convert the user input into vector
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return vectorizer, tfidf_matrix

# Perform self-match and add columns
def compute_self_similarity(df, description_col, code_col, tfidf_matrix):
    # finds the similarity between the two vectors and return the nested list of similarity score
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # looks into the similarity matrix to find the column of the highest score in each row and return an 1d array
    best_match_indices = similarity_matrix.argmax(axis=1)
    # finds the maximum score in each row and return an 1d array
    best_scores = similarity_matrix.max(axis=1)
    best_matches = [df.iloc[i][description_col] for i in best_match_indices]
    match_codes = [df.iloc[i][code_col] for i in best_match_indices]

    df['Matched Description'] = best_matches
    df['Matched Code'] = match_codes
    df['Similarity Score'] = best_scores
    return df

# Calculate evaluation metrics
def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Perform single user input match
def match_user_input(user_input, vectorizer, tfidf_matrix, df, description_col, code_col, cost_col, flag, threshold=0.5):
    # transform the user input into vector
    query_vec = vectorizer.transform([user_input])
    # calculate cosine similarity between the user input vector and the TF-IDF matrix and return the similarity scores for matching row
    sim_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    # find the index of the highest similarity score
    top_idx = sim_scores.argmax()
    # get the top score
    top_score = sim_scores[top_idx]

    if top_score < threshold:
        return None, None, None, top_score

    top_match = df.iloc[top_idx][description_col]
    top_code = df.iloc[top_idx][code_col]
    if flag == 'CCSR':
        top_cost = df.iloc[top_idx][cost_col]
        return top_match, top_code, top_cost, top_score
    else:return top_match, top_code, None, top_score

# ICD Support 
def match_icd_description(user_input,desc_col='SHORT DESCRIPTION (VALID ICD-10 FY2025)', code_col='CODE', threshold=0.5):
    icd_df = pd.read_excel("./data/icd/icd.xlsx")
    icd_df = icd_df.drop("NF EXCL", axis=1)
    vectorizer, tfidf_matrix = create_tfidf_matrix(icd_df[desc_col])
    return match_user_input(user_input, vectorizer, tfidf_matrix, icd_df, desc_col, code_col, cost_col=None, threshold=threshold, flag="ICD")

# Extract text from PDF and DOCX files
def extract_text_from_pdf(file):
    # Use fitz (PyMuPDF) to extract text from PDF
    doc = fitz.open(stream=file.read(), filetype="pdf")
    # Extract text from each para and join them into a single string
    return " ".join(page.get_text() for page in doc)

def extract_text_from_docx(file):
    # Use python-docx to extract text from DOCX
    doc = docx.Document(file)
    # Extract text from each paragraph and join them into a single string
    return " ".join(para.text for para in doc.paragraphs if para.text.strip())

def extract_possible_diagnoses(text):
    # Break big string into sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    # Heuristic: filter sentences likely to be diagnoses
    diagnosis_like = [s for s in sentences if 4 < len(s.split()) < 25]
    return diagnosis_like


