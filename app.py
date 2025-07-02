import os
import json
import streamlit as st
from datetime import datetime
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import requests

# File paths
data_dir = "data"
raw_text_path = os.path.join(data_dir, "raw_text.json")
corrections_path = os.path.join(data_dir, "corrections.json")

# Load raw text data
with open(raw_text_path, "r") as f:
    raw_text = json.load(f)

# Initialize ChromaDB with embedding function
embedding_func = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="rag_collection", embedding_function=embedding_func
)

# Add data to ChromaDB (if not already)
if len(collection.get()["ids"]) == 0:
    for i, doc in enumerate(raw_text):
        collection.add(documents=[str(doc)], ids=[f"doc_{i}"])

# Load corrections
def load_corrections():
    try:
        with open(corrections_path, "r") as f:
            return json.load(f)
    except:
        return []

# Save corrected answer
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

# RAG answer logic
def retrieve_and_answer(query):
    # First check manual corrections
    for entry in load_corrections():
        if entry["question"].strip().lower() == query.strip().lower():
            return entry["corrected_answer"]

    # ChromaDB top match
    results = collection.query(query_texts=[query], n_results=1)
    context = results["documents"][0][0]

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
        return f"Error: {e}"

# --- Streamlit UI ---

st.title("RAG Q&A Chatbot")

query = st.text_input("Ask your question:")
if st.button("Get Answer") and query:
    answer = retrieve_and_answer(query)
    st.write("### Answer")
    st.write(answer)

    with st.expander("Suggest a correction"):
        corrected = st.text_area("Corrected Answer")
        if st.button("Submit Correction") and corrected:
            save_correction(query, answer, corrected)
            st.success("Correction saved. Thank you!")
