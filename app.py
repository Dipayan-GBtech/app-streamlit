import os
import json
import requests
import streamlit as st
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import numpy as np

# --- Config ---
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
raw_text_path = os.path.join(data_dir, "raw_text.json")
corrections_path = os.path.join(data_dir, "corrections.json")
embedding_model = "togethercomputer/m2-bert-80M-8k-retrieval"  # fast & small
llm_model = "mistral-7b-instruct"

# --- Load raw text ---
with open(raw_text_path, "r") as f:
    raw_text = json.load(f)

# --- Embed documents using Together ---
@st.cache_data(show_spinner="Embedding documents...")
def embed_documents(texts):
    response = requests.post(
        "https://api.together.xyz/inference",
        headers={"Authorization": f"Bearer {st.secrets['TOGETHER_API_KEY']}"},
        json={
            "model": embedding_model,
            "input": texts
        }
    )
    return np.array(response.json()["embeddings"])

doc_embeddings = embed_documents([str(x) for x in raw_text])
nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(doc_embeddings)

# --- Correction file handlers ---
def load_corrections():
    try:
        with open(corrections_path, "r") as f:
            return json.load(f)
    except:
        return []

def save_correction(q, original, corrected):
    corrections = load_corrections()
    corrections.append({
        "question": q.strip(),
        "original_answer": original,
        "corrected_answer": corrected,
        "timestamp": datetime.now().isoformat()
    })
    with open(corrections_path, "w") as f:
        json.dump(corrections, f, indent=2)

# --- Answer retrieval ---
def retrieve_and_answer(query):
    # Return corrected answer if it exists
    for entry in load_corrections():
        if entry["question"].strip().lower() == query.strip().lower():
            return entry["corrected_answer"]

    # Embed query
    r = requests.post(
        "https://api.together.xyz/inference",
        headers={"Authorization": f"Bearer {st.secrets['TOGETHER_API_KEY']}"},
        json={"model": embedding_model, "input": [query]}
    )
    query_vec = np.array(r.json()["embeddings"])

    # Search
    _, indices = nn.kneighbors(query_vec)
    context = str(raw_text[indices[0][0]])

    # Ask LLM
    prompt = f"""
Answer the following question using the provided context:
Question: {query}
Context: {context}
Answer:
"""

    r = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": f"Bearer {st.secrets['TOGETHER_API_KEY']}"},
        json={
            "model": llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
    )
    return r.json()["choices"][0]["message"]["content"].strip()

# --- Streamlit Interface ---
st.title("💬 RAG Q&A (Together.ai only)")

query = st.text_input("Ask a question:")
if st.button("Get Answer") and query:
    answer = retrieve_and_answer(query)
    st.write("### Answer")
    st.write(answer)

    with st.expander("Suggest a correction"):
        correction = st.text_area("Corrected Answer")
        if st.button("Submit Correction") and correction:
            save_correction(query, answer, correction)
            st.success("✅ Correction saved!")
