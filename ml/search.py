from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    source: str # 'vector', 'keyword', 'graph'

class HybridSearchEngine:

    def __init__(self, embedding_generator, graph, storage=None):

        self.embedding_generator = embedding_generator
        self.graph = graph
        self.storage = storage
        self.chunks = []

    def index_chunk(self, chunk_id, text, vector, token_count=0):

        if self.storage:
            # Store in PostgreSQL
            self.storage.insert_chunk(
                chunk_id,
                text,
                token_count,
                vector
            )
        else:
            # fallback to in-memory
            self.chunks.append({
                "id": chunk_id,
                "text": text,
                "vector": vector,
                "token_count": token_count
            })

    def _vector_search(self, query_vec, top_k=5):

        if self.storage:
            # Use Postgres + pgvector
            rows = self.storage.search_vector(query_vec, limit=top_k)
            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "text": row[1],
                    "score": 1 - row[2]  # convert distance to similarity score
                })
            return results
        else:
            # fallback in-memory search
            results = []
            for chunk in self.chunks:
                dist = np.linalg.norm(np.array(chunk["vector"]) - np.array(query_vec))
                results.append({
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "score": 1 - dist
                })
            # sort by descending score
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

    def _keyword_search(self, query_str, top_k=5):

        results = []
        for chunk in self.chunks:
            score = sum(1 for word in query_str.split() if word.lower() in chunk["text"].lower())
            if score > 0:
                results.append({
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "score": float(score)
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


    def search(self, query_str, top_k=5):

        # 1. Vector search
        query_vec = self.embedding_generator.model.encode(query_str)
        vector_results = self._vector_search(query_vec, top_k=top_k)

        # 2. Keyword search
        keyword_results = self._keyword_search(query_str, top_k=top_k)

        # 3. Merge and Rerank using Reciprocal Rank Fusion
        merged = self._reciprocal_rank_fusion(vector_results, keyword_results)

        vector_ids = {r["id"] for r in vector_results}
        keyword_ids = {r["id"] for r in keyword_results}

        results = []
    
        for item in merged[:top_k]: # tag result for where it was found
            in_vec = item["id"] in vector_ids
            in_kw = item["id"] in keyword_ids
            source = "hybrid" if (in_vec and in_kw) else ("vector" if in_vec else "keyword") 
            results.append(SearchResult(
                chunk_id=item["id"],
                text=item["text"],
                score=item["score"],
                source=source
            ))
        return results


    def _reciprocal_rank_fusion(self, vector_results, keyword_results, k = 60):
        scores = {}
        texts = {}

        for rank, result in enumerate(vector_results, 1):
            scores[result['id']] = 1 / (k + rank) # equation on stage 6
            texts[result['id']] = result['text']

        for rank, result in enumerate(keyword_results, 1):
            span_id = result['id']
            scores[span_id] = scores.get(span_id, 0) + 1 / (k + rank)
            texts[span_id] = result['text']

        merged = [{"id": sid, "text": texts[sid], "score": score} for sid, score in scores.items()]
        merged.sort(key=lambda x: x["score"], reverse=True) # sort by descending score
        return merged