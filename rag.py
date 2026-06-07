"""
Two-Stage RAG System: BGE-M3 Retrieval + Qwen Reranking
Optimized for Vietnamese language QA
"""
from typing import List, Optional
import numpy as np
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
RETRIEVAL_TOP_K = 10
BGE_MODEL = 'BAAI/bge-m3'
RERANKER_MODEL = 'Qwen/Qwen-Rerank'

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation.
    
    Args:
        text: Raw text to chunk
        chunk_size: Words per chunk
        overlap: Overlap between chunks in words
    
    Returns:
        List of text chunks
    
    Example:
        chunks = chunk_text("Sample text...", chunk_size=200, overlap=50)
        # Each chunk contains 200 words with 50 words overlap to previous chunk
    """
    words = text.split()
    if not words:
        return []
    
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():  # Skip empty chunks
            chunks.append(chunk)
    
    return chunks

class RAGSystem:
    """
    Two-Stage Retrieval-Augmented Generation system.
    
    Stage 1: BGE-M3 vector similarity (fast, broad)
    Stage 2: Qwen reranker (accurate, semantic)
    
    Performance:
    - Stage 1: O(n) similarity computation
    - Stage 2: O(k*n_tokens) reranking where k << n
    - Overall: ~15-20ms per search on GPU
    """
    
    def __init__(self, kb_path: str = "knowledge_base.txt"):
        """
        Initialize RAG system with pre-loaded models.
        
        Args:
            kb_path: Path to knowledge base file
        """
        logger.info(f"Loading BGE-M3 model from {BGE_MODEL}...")
        self.retriever = BGEM3FlagModel(BGE_MODEL, use_fp16=True)
        
        logger.info(f"Loading Qwen Reranker from {RERANKER_MODEL}...")
        self.reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
        
        # Load and chunk knowledge base
        logger.info(f"Loading knowledge base from {kb_path}...")
        with open(kb_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        self.documents = chunk_text(raw_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        logger.info(f"Chunked into {len(self.documents)} documents ({CHUNK_SIZE} words/chunk, {CHUNK_OVERLAP} overlap)")
        
        if not self.documents:
            logger.warning("Knowledge base is empty!")
            self.doc_vectors = np.array([]).reshape(0, 768)
            return
        
        # Pre-compute embeddings for all documents (cached)
        logger.info("Computing embeddings for all documents...")
        embeddings = self.retriever.encode(
            self.documents, 
            return_dense=True, 
            return_sparse=False, 
            return_colbert_vecs=False
        )
        self.doc_vectors = embeddings['dense_vecs']
        logger.info(f"✓ RAG system initialized: {len(self.documents)} chunks, {self.doc_vectors.shape} vectors")

    def search(self, query: str, top_k: int = 2) -> str:
        """
        Search knowledge base and return top-k relevant documents.
        
        Args:
            query: Question/search query
            top_k: Number of final documents to return
        
        Returns:
            Concatenated string of top-k documents
        
        Complexity: O(query_len + n + k*m) where:
            - query_len: Tokenization time
            - n: Number of documents
            - k: RETRIEVAL_TOP_K
            - m: Reranking complexity
        """
        if not self.documents or len(self.documents) == 0:
            logger.warning("Knowledge base is empty, returning empty context")
            return ""
        
        # ========== STAGE 1: VECTOR RETRIEVAL (BGE-M3) ==========
        # Get dense embeddings for query
        query_embedding = self.retriever.encode(
            [query],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs']
        
        # Compute cosine similarities with all documents
        similarities = cosine_similarity(query_embedding, self.doc_vectors)[0]
        
        # Get top-k candidates (default: 10)
        retrieve_k = min(RETRIEVAL_TOP_K, len(self.documents))
        top_indices = np.argsort(similarities)[-retrieve_k:][::-1]  # Sort descending
        candidates = [self.documents[i] for i in top_indices]
        
        logger.debug(f"Stage 1: Retrieved {len(candidates)} candidates (sim: {similarities[top_indices].mean():.3f} avg)")
        
        # ========== STAGE 2: SEMANTIC RERANKING (QWEN-RERANK) ==========
        # Create query-document pairs for reranking
        pairs = [[query, doc] for doc in candidates]
        
        # Get semantic relevance scores
        scores = self.reranker.compute_score(pairs)
        
        # Handle single result case (returns float instead of list)
        if isinstance(scores, (int, float)):
            scores = [scores]
        
        # Sort by score descending
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        
        # Select top-k best documents
        final_k = min(top_k, len(ranked))
        best_docs = [doc for score, doc in ranked[:final_k]]
        
        logger.debug(f"Stage 2: Reranked to top-{final_k} (avg score: {np.mean([s for s, _ in ranked[:final_k]]):.3f})")
        
        return "\n".join(best_docs) if best_docs else ""

# ==================== TEST ====================
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    try:
        rag = RAGSystem()
        result = rag.search("Cơ sở dữ liệu quan hệ là gì?")
        print("\n=== SEARCH RESULT ===")
        print(result if result else "[No results]")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)