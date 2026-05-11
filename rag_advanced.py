import os
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import warnings

# Suppress annoying huggingface warnings for clean output
warnings.filterwarnings("ignore")

# --- 1. SETUP & LOAD DATA ---
print("1. Waking up systems and loading data...")

# Connect to Chroma and fetch ALL existing chunks to train our Keyword engine
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_collection("aegis_papers")

all_data = collection.get(include=['documents', 'metadatas'])
all_chunks = all_data['documents']

# --- 2. INITIALIZE ENGINES ---
print("2. Initializing BM25 Keyword Engine and Cross-Encoder Judge...")

# A. The Keyword Engine (BM25)
# It needs the text split into raw lowercase words
tokenized_corpus = [doc.lower().split(" ") for doc in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)

# B. The Re-ranker (Cross-Encoder)
# This model is specifically trained to score how well a document answers a query
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)


# --- 3. THE HYBRID SEARCH ALGORITHM (RRF) ---
def hybrid_search(query, top_k=10):
    """Combines Vector Search and Keyword Search using Reciprocal Rank Fusion."""
    
    # 1. Get Vector Results (Semantic Meaning)
    vector_res = collection.query(query_texts=[query], n_results=top_k)
    vector_docs = vector_res['documents'][0]
    
    # 2. Get BM25 Results (Exact Keywords)
    tokenized_query = query.lower().split(" ")
    bm25_docs = bm25.get_top_n(tokenized_query, all_chunks, n=top_k)
    
    # 3. Reciprocal Rank Fusion (RRF)
    # The math formula: score = 1 / (k + rank)
    rrf_scores = {}
    
    # Score the vector docs
    for rank, doc in enumerate(vector_docs):
        rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (60 + rank + 1)
        
    # Score the BM25 docs
    for rank, doc in enumerate(bm25_docs):
        rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (60 + rank + 1)
        
    # Sort everything by their combined RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return just the text of the top_k winners
    return [doc[0] for doc in sorted_docs[:top_k]]


# --- 4. THE RE-RANKER ---
def rerank_results(query, candidate_docs, top_k=3):
    """Uses a Cross-Encoder to act as the final judge and re-score the candidates."""
    
    # Create pairs of [Query, Document] for the model to judge
    pairs = [[query, doc] for doc in candidate_docs]
    
    # The model predicts a relevancy score for each pair
    scores = reranker.predict(pairs)
    
    # Zip the docs and scores together, then sort highest to lowest
    scored_docs = list(zip(candidate_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc[0] for doc in scored_docs[:top_k]]


# --- 5. THE TEST ARENA ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print(" PROJECT AEGIS - ADVANCED RAG PIPELINE ")
    print("="*60 + "\n")
    
    # We use a query with a highly specific keyword to test the system
    test_query = "What data did Sui et al. (2025) use?"
    print(f"QUERY: '{test_query}'\n")
    print("-" * 60)
    
    # Test A: Pure Vector (Chroma Only)
    vector_res = collection.query(query_texts=[test_query], n_results=1)['documents'][0][0]
    print(f"[PURE VECTOR] Top Result:\n...{vector_res[:150]}...\n")
    
    # Test B: Pure Keyword (BM25 Only)
    bm25_res = bm25.get_top_n(test_query.lower().split(" "), all_chunks, n=1)[0]
    print(f"[PURE BM25] Top Result:\n...{bm25_res[:150]}...\n")
    
    # Test C: Hybrid + Re-ranking (The Industry Standard)
    hybrid_candidates = hybrid_search(test_query, top_k=10) # Cast a wide net
    final_winners = rerank_results(test_query, hybrid_candidates, top_k=1) # Pick the absolute best
    
    print(f"[HYBRID + RERANK] Top Result:\n...{final_winners[0][:150]}...\n")
    print("="*60)
