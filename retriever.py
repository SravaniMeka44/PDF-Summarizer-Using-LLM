# retriever.py
from embeddings import Embedder

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.embedder = Embedder()

    def retrieve(self, query, top_k=3):
        """
        Accepts a query string, embeds it, searches vector store,
        returns the top_k most relevant documents.
        """
        query_embedding = self.embedder.embed([query])[0]  # get first embedding vector
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return results