import os
import json
import numpy as np
import streamlit as st
import faiss
import requests
from datetime import datetime
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load Together API key from secrets
#API_KEY = st.secrets["TOGETHER_API_KEY"]
API_KEY = st.secrets["OPENROUTER_API_KEY"]

# Paths
DATA_DIR = "data"
RAW_TEXT_PATH = os.path.join(DATA_DIR, "raw_text.json")
CORRECTIONS_PATH = os.path.join(DATA_DIR, "corrections.json")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

# Load raw text
with open(RAW_TEXT_PATH, "r") as f:
    raw_text = json.load(f)

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Build or load FAISS index (persistent across runs)
@st.cache_resource
def build_or_load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    else:
        embeddings = model.encode([str(entry) for entry in raw_text])
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, INDEX_PATH)
        return index


index = build_or_load_index()

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

def extract_structured_json(answer_text):
    """
    Uses the same LLM to extract a structured JSON from the answer.
    """
    extraction_prompt = f"""You are an expert in parsing healthcare benefit answers.
From the following answer, extract ONLY the following fields if mentioned:
- plan_paid
- providers_responsibility
- copay
- coinsurance
- deductible
- employee_responsibility

If a field is not mentioned, omit it or set it to null.
Return a valid JSON object with only these keys. Do not add explanations.

Answer:
{answer_text}

JSON:"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                #"model": "openai/gpt-oss-120b",
                "model": "qwen/qwen-2.5-72b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a precise JSON extractor."},
                    {"role": "user", "content": extraction_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 256
            }
        )
        if response.status_code == 200:
            raw_output = response.json()["choices"][0]["message"]["content"].strip()
            # Try to extract JSON from possible markdown or text
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:]  # remove ```json
            if raw_output.endswith("```"):
                raw_output = raw_output[:-3]
            # Parse JSON
            import json
            parsed = json.loads(raw_output)
            return parsed
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Failed to extract JSON: {str(e)}"}

# Generate answer from Together AI (Qwen model)

def generate_with_ollama(prompt):
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                },
            json={
                #"model": "openai/gpt-oss-120b",  # OpenRouter model ID for Qwen2.5-14B-Instruct
                "model": "qwen/qwen-2.5-72b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 256
            }
        )
        # Handle potential errors in response
        if response.status_code != 200:
            return f"⚠️ API Error ({response.status_code}): {response.text}"
        
        return response.json()["choices"][0]["message"]["content"].strip()
    
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"


# Retrieval + RAG
def retrieve_and_answer(query):
    # Check for corrected answer (exact match correction)
    corrections = load_corrections()
    for entry in corrections:
        if entry["question"].strip().lower() == query.strip().lower():
            return entry["corrected_answer"], None  # No context if correction used

    # Encode the query
    try:
        query_emb = model.encode([query])
    except Exception as e:
        st.error(f"❌ Error encoding query: {e}")
        return "⚠️ Unable to process your question.", None

    # Search FAISS index
    try:
        D, I = index.search(np.array(query_emb), k=1)
    except Exception as e:
        st.error(f"❌ Error during FAISS search: {e}")
        return "⚠️ Retrieval system is not working properly.", None

    # --- Safety checks ---
    # 1. Check if search returned any valid index
    if I is None or len(I) == 0 or len(I[0]) == 0:
        return "I don't know.", None

    retrieved_index = I[0][0]

    # 2. Ensure the index is within the bounds of raw_text
    if retrieved_index < 0 or retrieved_index >= len(raw_text):
        st.warning(f"⚠️ Invalid index returned by FAISS: {retrieved_index}")
        return "I don't know.", None

    # 3. Double-check that raw_text is a list and not empty
    if not isinstance(raw_text, list) or len(raw_text) == 0:
        st.error("❌ raw_text is not a valid list or is empty.")
        return "I don't know.", None

    # All checks passed — retrieve context
    try:
        context = str(raw_text[retrieved_index])
    except Exception as e:
        st.error(f"❌ Error accessing raw_text[{retrieved_index}]: {e}")
        return "I don't know.", None

    # Build prompt for LLM
    prompt = f"""Use the context below to answer the question clearly and concisely.
If the context does not contain relevant information, say "I don't know."

Context:
{context}

Question: {query}

Answer:"""

    # Generate answer
    try:
        answer = generate_with_ollama(prompt)
    except Exception as e:
        st.error(f"❌ Error generating answer: {e}")
        answer = "⚠️ Sorry, I couldn't generate a response."

    return answer, context
# --- Streamlit UI ---
# --- Streamlit UI ---
st.set_page_config(page_title="Q&A", page_icon="🤖")
st.title("🔎 Q&A ")

query = st.text_input("Ask your question:")

debug = st.checkbox("Show debug info (context, prompt)")

if st.button("Get Answer") and query:
    answer, context = retrieve_and_answer(query)
    st.session_state["query"] = query
    st.session_state["answer"] = answer
    st.session_state["context"] = context

    # Extract structured JSON from the answer
    structured_data = extract_structured_json(answer)
    st.session_state["structured_json"] = structured_data

    # Display original answer
    st.markdown("### ✅ Answer")
    st.write(answer)

    # Display structured JSON
    st.markdown("### 📦 Structured Data (JSON)")
    st.json(structured_data)

    if debug:
        st.markdown("### 📄 Retrieved Context")
        st.write(context)

        st.markdown("### 🤖 Prompt Sent to LLM")
        st.code(f"""Use the context below to answer the question clearly and concisely.
If the context does not contain relevant information, say "I don't know."

Context:
{context}

Question: {query}

Answer:""")

# Feedback section remains unchanged
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
