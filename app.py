import json
import numpy as np
import os
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from datetime import datetime
import requests

# Load secrets
API_KEY = st.secrets["019f4b463649d0f2cb6e13198e8b7547974c3de9e7442f56145400d843661be7"]

# Paths
data_dir = "data"
raw_text_path = os.path.join(data_dir, "raw_text.json")
corrections_path = os.path.join(data_dir, "corrections.json")

# Load raw_text
with open(raw_text_path, "r") as f:
    raw_text = json.load(f)

# Load model and build FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([str(entry) for entry in raw_text])
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

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

def generate_with_mistral_together(prompt):
    try:
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating response: {e}"

def retrieve_and_answer(query):
    corrections = load_corrections()
    for entry in corrections:
        if entry["question"].strip().lower() == query.strip().lower():
            return entry["corrected_answer"]

    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), k=1)
    context = str(raw_text[I[0][0]])

    prompt = f"""Answer the following question using the provided context:
Question: {query}
Context:
{context}
Answer:"""

    return generate_with_mistral_together(prompt)

# Streamlit UI
st.set_page_config(page_title="RAG QA", page_icon="🤖")
st.title("🔍 RAG Q&A with Mistral (Together AI)")

query = st.text_input("Ask a question:")
if st.button("Get Answer") and query:
    answer = retrieve_and_answer(query)
    st.session_state["last_answer"] = answer
    st.session_state["last_query"] = query
    st.write("### Answer")
    st.write(answer)

if "last_answer" in st.session_state:
    st.markdown("---")
    st.subheader("Was this answer correct?")
    col1, col2 = st.columns(2)
    if col1.button("Yes"):
        st.success("Thanks for your feedback!")
    if col2.button("No"):
        corrected = st.text_area("Suggest a correction:", value=st.session_state["last_answer"])
        if st.button("Submit Correction"):
            save_correction(
                st.session_state["last_query"],
                st.session_state["last_answer"],
                corrected
            )
            st.success("Correction submitted.")
