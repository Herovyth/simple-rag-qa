from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, retrieved_chunks, top_k=5):
        if not retrieved_chunks:
            return []

        pairs = [[query, text] for text, _ in retrieved_chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(retrieved_chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [text for ((text, _), _) in ranked[:top_k]]
