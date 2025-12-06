import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
import os

# Initialize OpenAI client once with your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedder class using SentenceTransformer
class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

# Simple VectorStore for embeddings and docs
class VectorStore:
    def __init__(self):
        self.embeddings = []
        self.docs = []
    def add(self, emb, doc):
        self.embeddings.append(emb)
        self.docs.append(doc)
    def search(self, query_emb, top_k=3):
        all_emb = np.array(self.embeddings)
        dot = np.dot(all_emb, query_emb)
        norms = np.linalg.norm(all_emb, axis=1) * np.linalg.norm(query_emb)
        sims = dot / norms
        top_idxs = np.argsort(sims)[-top_k:][::-1]
        return [(self.docs[i], sims[i]) for i in top_idxs]

# Extract text from PDF
def extract_text(file):
    with pdfplumber.open(file) as pdf:
        texts = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        return "\n\n".join(texts)

# Build prompt not needed explicitly as we will pass messages in chat format

# Query OpenAI chat completion
def query_openai(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Or "gpt-4", "gpt-3.5-turbo" etc.
        messages=messages,
        max_tokens=500,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

# --- Streamlit UI ---

st.title("📄 PDF Upload + Multi-turn LLM Q&A Chat")

# Initialize session state variables
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embedder" not in st.session_state:
    st.session_state.embedder = Embedder()
if "chat_history" not in st.session_state:
    # list of dicts with roles and contents
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_file is not None:
    if not st.session_state.doc_chunks:
        with st.spinner("Extracting text and processing document..."):
            full_text = extract_text(uploaded_file)
            chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]
            st.session_state.doc_chunks = chunks
            
            embeddings = st.session_state.embedder.embed(chunks)
            
            store = VectorStore()
            for emb, doc in zip(embeddings, chunks):
                store.add(emb, doc)
            st.session_state.vector_store = store
            
            st.success("Document processed and ready for questions!")

    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Searching for relevant context..."):
            query_emb = st.session_state.embedder.embed([question])[0]
            results = st.session_state.vector_store.search(query_emb, top_k=3)
            contexts = [doc for doc, score in results]

        # Append user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Build messages list: system prompt + context + chat history
        system_message = {
            "role": "system",
            "content": "You are a helpful AI assistant. Use the provided context to answer questions."
        }
        context_message = {
            "role": "system",
            "content": "Context:\n\n" + "\n\n---\n\n".join(contexts)
        }
        messages = [system_message, context_message] + st.session_state.chat_history

        with st.spinner("Generating answer..."):
            answer = query_openai(messages)

        # Append assistant answer to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Display conversation history
    if st.session_state.chat_history:
        st.markdown("### Conversation History:")
        for i in range(0, len(st.session_state.chat_history), 2):
            user_msg = st.session_state.chat_history[i]["content"]
            assistant_msg = ""
            if i+1 < len(st.session_state.chat_history):
                assistant_msg = st.session_state.chat_history[i+1]["content"]
            st.markdown(f"**You:** {user_msg}")
            st.markdown(f"**Assistant:** {assistant_msg}")
            st.write("---")

    # Display context used for last answer
    if question:
        st.markdown("### Context used for last answer:")
        for i, (doc, score) in enumerate(results, 1):
            st.write(f"{i}. (score: {score:.3f}) {doc[:300]}...")