import os
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# --- 1. INITIALIZE FASTAPI & AI ---
app = FastAPI(title="Project Aegis Backend")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Loading Database and Engines...")
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_collection("aegis_papers")

all_data = collection.get(include=['documents', 'metadatas'])
all_chunks = all_data['documents']

# Setup Hybrid Search Engines
tokenized_corpus = [doc.lower().split(" ") for doc in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

# --- 2. DATA MODELS ---
class QueryRequest(BaseModel):
    question: str
    chat_history: str = "" # We pass memory as a string for simplicity in the MVP

# --- 3. CORE LOGIC ---
def advanced_retrieval(query, top_k=5):
    # Vector
    vector_res = collection.query(query_texts=[query], n_results=top_k)['documents'][0]
    # BM25
    bm25_res = bm25.get_top_n(query.lower().split(" "), all_chunks, n=top_k)
    
    # RRF Merge
    rrf_scores = {}
    for rank, doc in enumerate(vector_res + bm25_res):
        rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (60 + rank + 1)
    
    hybrid_docs = [doc[0] for doc in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:10]]
    
    # Rerank
    pairs = [[query, doc] for doc in hybrid_docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(list(zip(hybrid_docs, scores)), key=lambda x: x[1], reverse=True)
    
    return [doc[0] for doc in scored_docs[:3]] # Return top 3 verified chunks

# --- 4. THE API ENDPOINT ---
@app.post("/query")
async def handle_query(req: QueryRequest):
    print(f"Received query: {req.question}")
    
    # 1. Retrieve
    best_chunks = advanced_retrieval(req.question)
    context_str = "\n\n".join(best_chunks)
    
   # 2. Build Prompt with Memory
    sys_prompt = (
        "You are the Project Aegis AI. Answer ONLY using the provided context. "
        "Cite the authors/studies mentioned. Do NOT include any XML tags (like <context> or <question>) "
        "in your final output. Just write the plain text answer."
    )
    user_prompt = f"""<chat_history>\n{req.chat_history}\n</chat_history>
    
<context>\n{context_str}\n</context>

<question>\n{req.question}\n</question>"""

    # 3. Generate
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.1
    )
    
    return {"answer": response.choices[0].message.content, "chunks_used": len(best_chunks)}
