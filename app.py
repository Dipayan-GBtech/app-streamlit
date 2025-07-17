import os
import json
import numpy as np
import streamlit as st
import faiss
import requests
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Load Together API key from secrets
API_KEY = st.secrets["TOGETHER_API_KEY"]

# Paths
DATA_DIR = "data"
RAW_TEXT_PATH = os.path.join(DATA_DIR, "raw_text.json")
CORRECTIONS_PATH = os.path.join(DATA_DIR, "corrections.json")

# Load raw text
with open(RAW_TEXT_PATH, "r") as f:
    raw_text = json.load(f)

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Build FAISS index
@st.cache_resource
def build_index():
    embeddings = model.encode([str(entry) for entry in raw_text])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

index = build_index()

# Load corrections
def load_corrections():
    try:
        with open(CORRECTIONS_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# Save correction
def save_correction(question, original_answer, corrected_answer):
    corrections = load_corrections()
    corrections.append({
        "question": question.strip(),
        "original_answer": original_answer,
        "corrected_answer": corrected_answer,
        "timestamp": datetime.now().isoformat()
    })
    with open(CORRECTIONS_PATH, "w") as f:
        json.dump(corrections, f, indent=2)
    print("✅ Correction saved")

# Generate answer from Together AI (Mistral model)
def generate_with_together(prompt):
    try:
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                #"model": "mistralai/Mistral-7B-Instruct-v0.2",
                "model": "Qwen/Qwen1.5-0.5B-Chat",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"

# Retrieval + RAG
def retrieve_and_answer(query):
    # Check for corrected answer
    corrections = load_corrections()
    for entry in corrections:
        if entry["question"].strip().lower() == query.strip().lower():
            return entry["corrected_answer"]

    # RAG logic
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), k=1)
    context = str(raw_text[I[0][0]])

    prompt = f"""Answer the following question using the provided context:
Question: {query}
Context:
{context}
Answer:"""

    return generate_with_together(prompt)

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Q&A (Together)", page_icon="🤖")
st.title("🔎 RAG-based Q&A with Together AI (Mistral)")

query = st.text_input("Ask your question:")

if st.button("Get Answer") and query:
    answer = retrieve_and_answer(query)
    st.session_state["query"] = query
    st.session_state["answer"] = answer
    st.markdown("### ✅ Answer")
    st.write(answer)

if "answer" in st.session_state:
    st.markdown("---")
    st.subheader("Was this answer correct?")
    col1, col2 = st.columns(2)

    if col1.button("👍 Yes"):
        st.success("Thanks for your feedback!")

    if col2.button("👎 No"):
        corrected = st.text_area("Suggest a better answer:", value=st.session_state["answer"])
        if st.button("Submit Correction"):
            save_correction(
                st.session_state["query"],
                st.session_state["answer"],
                corrected
            )
            st.success("Correction saved successfully.")
