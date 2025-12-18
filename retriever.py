from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import torch


class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks

        # BM25
        self.tokenized_chunks = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

        # Semantic
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunk_embeddings = self.embedder.encode(
            chunks, convert_to_tensor=True
        )

    def bm25_search(self, query, top_k=10):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def semantic_search(self, query, top_k=10):
        q_emb = self.embedder.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, self.chunk_embeddings)[0]

        top_idx = torch.topk(scores, k=top_k).indices
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def retrieve(self, query, mode="both", top_k=10):
        if mode == "off":
            return []

        results = []

        if mode in ("bm25", "both"):
            results.extend(self.bm25_search(query, top_k))

        if mode in ("semantic", "both"):
            results.extend(self.semantic_search(query, top_k))

        unique = {}
        for text, score in results:
            unique[text] = max(score, unique.get(text, -1))

        return list(unique.items())
