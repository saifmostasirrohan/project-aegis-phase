import os
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# --- SETUP ---
load_dotenv()
app = FastAPI(title="Project Aegis - RAG API")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="aegis_papers")

# --- DATA MODELS ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    # Chroma uses L2 distance. Lower is better. A distance > 1.5 usually means "completely unrelated"
    distance_threshold: float = 1.5 

# --- ENDPOINTS ---
@app.post("/query")
async def query_aegis(request: QueryRequest):
    print(f"\n--- NEW QUERY RECEIVED: {request.question} ---")
    
    # 1. RETRIEVAL (Now explicitly asking for distances)
    results = collection.query(
        query_texts=[request.question],
        n_results=request.top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = results['documents'][0]
    metadata = results['metadatas'][0]
    distances = results['distances'][0]
    
    # 2. THE THRESHOLD CHECK (Don't waste LLM tokens on bad questions)
    best_distance = distances[0] if distances else 999.0
    print(f"Best chunk distance score: {best_distance}")
    
    if best_distance > request.distance_threshold:
        print("Query out of scope. Rejecting before LLM call.")
        return {
            "answer": "I cannot find relevant information in the provided research papers.",
            "sources": [],
            "status": "rejected_by_threshold"
        }
        
    # 3. PROMPT CONSTRUCTION
    context_string = "\n\n".join([f"--- CHUNK ---\n{chunk}" for chunk in chunks])
    
    system_prompt = (
        "You are an expert AI research assistant for Project Aegis. "
        "Answer ONLY using the provided <context>. If the answer is not in the context, "
        "explicitly say 'I cannot find this information'. Do not hallucinate."
    )
    
    user_prompt = f"<context>\n{context_string}\n</context>\n\n<question>\n{request.question}\n</question>"
    
    # 4. LLM GENERATION
    print("Generating answer via Groq...")
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.1,
    )
    
    # 5. RESPONSE FORMATTING
    # Format sources neatly for the JSON response
    sources_list = [
        {"paper": md.get("source", "Unknown"), "page": md.get("page", "Unknown")} 
        for md in metadata
    ]
    
    return {
        "answer": response.choices[0].message.content,
        "sources": sources_list,
        "status": "success"
    }
