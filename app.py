import json
import numpy as np
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from datetime import datetime
import requests
from sklearn.neighbors import NearestNeighbors

# Paths
data_dir = "data"
raw_text_path = os.path.join(data_dir, "raw_text.json")
corrections_path = os.path.join(data_dir, "corrections.json")

# Load raw_text
with open(raw_text_path, "r") as f:
    raw_text = json.load(f)

# Load model and encode
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([str(entry) for entry in raw_text], convert_to_numpy=True)

# Build scikit-learn index using cosine distance
index = NearestNeighbors(n_neighbors=1, metric="cosine")
index.fit(embeddings)

def load_corrections():
    try:
        with open(corrections_path, "r") as f:
            return json.load(f)
    except:
        return []

def save_correction(question, original_answer, corrected_answer):
    corrections = load_corrections()
    corrections.append({
        "question": question.strip(),
        "original_answer": original_answer,
        "corrected_answer": corrected_answer,
        "timestamp": datetime.now().isoformat()
    })
    with open(corrections_path, "w") as f:
        json.dump(corrections, f, indent=2)

def retrieve_and_answer(query):
    # Check for manual corrections
    corrections = load_corrections()
    for entry in corrections:
        if entry["question"].strip().lower() == query.strip().lower():
            return entry["corrected_answer"]

    # Retrieve most relevant context using cosine similarity
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.kneighbors(query_emb, return_distance=True)
    context = str(raw_text[I[0][0]])

    # Prompt
    prompt = f"""
Answer the following question using the provided context:
Question: {query}
Context:
{context}
Answer:
"""

    try:
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",  # replace with your model provider if needed
            headers={"Authorization": f"Bearer {st.secrets['TOGETHER_API_KEY']}"},
            json={
                "model": "mistral-7b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
        )
        ans = response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        ans = f"Error generating answer: {e}"

    return ans

# Streamlit UI
st.title("RAG-based Q&A with Feedback")

query = st.text_input("Ask a question:")
if st.button("Get Answer") and query:
    answer = retrieve_and_answer(query)
    st.write("### Answer")
    st.write(answer)

    with st.expander("Submit a correction"):
        corrected = st.text_area("Corrected Answer")
        if st.button("Submit Correction") and corrected:
            save_correction(query, answer, corrected)
            st.success("Correction submitted.")



