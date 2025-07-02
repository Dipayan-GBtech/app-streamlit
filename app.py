import os
import json
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from InstructorEmbedding import INSTRUCTOR
import requests

# Paths
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
raw_text_path = os.path.join(data_dir, "raw_text.json")
corrections_path = os.path.join(data_dir, "corrections.json")

# Load documents
with open(raw_text_path, "r") as f:
    raw_text = json.load(f)

# Load embedding model (NO HF auth needed)
model = INSTRUCTOR("hkunlp/instructor-base")
instruction = "Represent the document for retrieval:"

# Encode documents
doc_embeddings = [model.encode([[instruction, str(d)]])[0] for d in raw_text]

# Build NearestNeighbors index
nn = NearestNeighbors(n_neighbors=1, metric="cosine")
nn.fit(doc_embeddings)

# Load previous corrections
def load_corrections():
    try:
        with open(corrections_path, "r") as f:
            return json.load(f)
    except:
        return []

# Save human-corrected answer
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

# Retrieve + generate answer
def retrieve_and_answer(query):
    # If previously corrected, return that
    for entry in load_corrections():
        if entry["question"].strip().lower() == query.strip().lower():
            return entry["corrected_answer"]

    # Embed query and find closest doc
    query_vec = model.encode([[instruction, query]])
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
        return f"❌ Error generating answer: {e}"

# --- Streamlit Interface ---
st.title("🧠 RAG Q&A (No Hugging Face)")

question = st.text_input("Ask your question:")
if st.button("Get Answer") and question:
    answer = retrieve_and_answer(question)
    st.write("### Answer")
    st.write(answer)

    with st.expander("Suggest a correction"):
        corrected = st.text_area("Corrected Answer")
        if st.button("Submit Correction") and corrected:
            save_correction(question, answer, corrected)
            st.success("✅ Correction saved. Thank you!")
