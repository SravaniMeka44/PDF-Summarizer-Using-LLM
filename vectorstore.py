# vectorstore.py
import numpy as np

class VectorStore:
    def __init__(self):
        self.embeddings = []   # list of vectors
        self.documents = []    # list of original texts

    def add(self, embedding, document):
        """
        Stores one embedding + document.
        """
        self.embeddings.append(embedding)
        self.documents.append(document)

    def search(self, query_embedding, top_k=3):
        """
        Returns top_k most similar documents to the query embedding.
        """

        # Convert stored embeddings to numpy array for vector math
        all_embeddings = np.array(self.embeddings)

        # Compute cosine similarity
        dot_product = np.dot(all_embeddings, query_embedding)
        norms = np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarity_scores = dot_product / norms

        # Get top-k indices
        top_k_indices = np.argsort(similarity_scores)[-top_k:][::-1]

        # Return matched documents + similarity
        results = []
        for i in top_k_indices:
            results.append({
                "document": self.documents[i],
                "score": float(similarity_scores[i])
            })

        return results