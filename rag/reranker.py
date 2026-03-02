from sentence_transformers import SentenceTransformer, util

class SimpleReranker:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def rerank(self, query, docs):
        texts = [d["text"] for d in docs]
        q_emb = self.model.encode(query, normalize_embeddings=True)
        d_emb = self.model.encode(texts, normalize_embeddings=True)

        scores = util.cos_sim(q_emb, d_emb)[0]
        scored = list(zip(scores, docs))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [d for _, d in scored]