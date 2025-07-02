import os
import json
import numpy as np
import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import requests

# --- Load data ---
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
raw_text_path = os.path.join(data_dir, "raw_text.json")
corrections_path = os.path.join(data_dir, "corrections.json")

# Load documents
with open(raw_text_path, "r") as f:
    raw_text = json.load(f)

# Load model + encode docs
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode([str(d) for d in raw_text])
nn = NearestNeighbors(n_neighbors=1, metric="cosine")
nn.fit(doc_embeddings)

# --- Feedback tracking ---
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

# --- RAG pipeline ---
def retrieve_and_answer(query):
    # Return correction if exists
    for entry in load_corrections():
        if entry["question"].strip().lower() == query.strip().lower():
            return entry["corrected_answer"]

    # Embed and retrieve context
    query_vec = model.encode([query])
    _, indices = nn.kneighbors(query_vec)
    context = str(raw_text[indices[0][0]])

    prompt = f"""
Answer the following question using the provided context:
Question: {query}
Context: {context}
Answer:
"""

    try:
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {st.secrets['TOGETHER_API_KEY']}"},
            json={
                "model": "mistral-7b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error calling LLM: {e}"

# --- Streamlit UI ---
st.title("🔍 RAG Q&A System (Sklearn-based)")
question = st.text_input("Ask a question:")

if st.button("Get Answer") and question:
    answer = retrieve_and_answer(question)
    st.write("### Answer")
    st.write(answer)

    with st.expander("Suggest a correction"):
        correction = st.text_area("Corrected Answer")
        if st.button("Submit Correction") and correction:
            save_correction(question, answer, correction)
            st.success("✅ Correction saved. Thank you!")
