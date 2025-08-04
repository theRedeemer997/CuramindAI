# 🏥 CuramindAI

This project aims to help **medical coders** map **diagnosis descriptions** to standardized **CCSR codes** using **natural language processing (NLP)** techniques. It leverages **TF-IDF vectorization** and **cosine similarity** to find the best match for a diagnosis description and assign the most relevant medical code.

---

## 📦 Dataset

- The dataset contains **discharge summaries** with fields such as:
  - `CCSR Diagnosis Description` (free-text input)
  - `CCSR Diagnosis Code` (standardized medical code)
  - Other patient and hospital-related metadata

---

## 🧠 Project Workflow

1. **Data Preprocessing**
   - Drop irrelevant columns (e.g., payment types, birth weight).
   - Fill missing values (e.g., missing zip codes with `"000"`).
   - Filter to keep important columns for coding.

2. **Extract Unique Descriptions**
   - Unique descriptions from `CCSR Diagnosis Description` are extracted as reference corpus.

3. **TF-IDF Vectorization**
   - Apply `TfidfVectorizer` to convert descriptions into vector space.
   - Compute a similarity matrix between all entries and unique descriptions.

4. **Cosine Similarity Matching**
   - Identify the closest match (highest cosine similarity) for each diagnosis.
   - Map the best match back to the corresponding CCSR code.

5. **Add Results to DataFrame**
   - `Matched Description`
   - `Matched Code`
   - `Similarity Score`

6. **Export Output**
   - Save final results to `final_matched_results.csv`.

---

## 🔍 Example

| Input Description                | Matched Description         | Code   | Similarity |
|----------------------------------|------------------------------|--------|------------|
| Infection of the urinary tract  | Urinary tract infections     | GEN004 | 0.99       |
| Heart condition and fatigue     | Chronic heart failure        | CIR007 | 0.87       |

---

## 📖 What is a CCSR Code?

- **CCSR = Clinical Classifications Software Refined**
- Groups detailed ICD-10-CM diagnosis codes into ~531 clinically meaningful categories.
- Makes diagnosis data easier to analyze, report, and model.

Example:

| ICD-10 Code | Description                     | CCSR Code | CCSR Category            |
|-------------|---------------------------------|-----------|--------------------------|
| N39.0       | Urinary tract infection         | GEN004    | Urinary tract infections |
| I50.9       | Heart failure, unspecified      | CIR007    | Chronic heart failure    |

---

## 🛠 Technologies Used

- Python
- Pandas
- scikit-learn
  - `TfidfVectorizer`
  - `cosine_similarity`
- Jupyter Notebook

---

## 📁 Output

- Final results stored in:  
  `./data/final_matched_results.csv`

---

## ✅ Use Case

This tool can assist:
- Medical coders
- Health data scientists
- Insurance billing teams
- Hospitals performing clinical analytics

---

## 📌 Future Enhancements

- Add support for ICD-10-PCS (Procedure Coding System)
- Integrate with a Streamlit interface for live coding
- Improve matching using BERT-based embeddings

