import os
import chromadb
from groq import Groq
from dotenv import load_dotenv

# --- 1. INITIALIZATION ---
load_dotenv()
print("1. Waking up the systems...")

# Initialize Groq (The Brain) and Chroma (The Memory)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Connect to the exact collection you built in CP-03
collection = chroma_client.get_collection(name="aegis_papers")


# --- 2. THE RETRIEVAL ENGINE ---
def retrieve_context(query, top_k=3):
    """Searches Chroma vector database for the most relevant paper chunks."""
    print(f"2. Searching database for: '{query}'")
    
    # Chroma converts the query to math and finds the closest matches
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    # Extract the text chunks and their metadata (page numbers/sources)
    retrieved_chunks = results['documents'][0]
    retrieved_metadata = results['metadatas'][0]
    
    return retrieved_chunks, retrieved_metadata


# --- 3. PROMPT CONSTRUCTION (XML TAGGING) ---
def build_prompt(query, chunks):
    """Fences the retrieved chunks inside XML tags to prevent hallucinations."""
    print("3. Assembling the context-aware prompt...")
    
    # Combine the retrieved chunks into one big string
    context_string = "\n\n".join([f"--- CHUNK ---\n{chunk}" for chunk in chunks])
    
    # The System Prompt acts as the strict rules of engagement
    system_prompt = (
        "You are an expert AI research assistant for Project Aegis. "
        "Your job is to answer the user's question ONLY using the provided <context>. "
        "If the answer is not contained in the <context>, you must explicitly state: "
        "'I cannot find this information in the provided research papers.' "
        "Do not hallucinate or use outside knowledge."
    )
    
    # The User Prompt contains the actual data and the question
    user_prompt = f"""<context>
{context_string}
</context>

<question>
{query}
</question>"""

    return system_prompt, user_prompt


# --- 4. LLM GENERATION ---
def generate_answer(system_prompt, user_prompt):
    """Sends the strict prompt to Groq/Llama 3."""
    print("4. Sending prompt to Llama 3 on Groq...")
    
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.1-8b-instant", 
        temperature=0.1,  # Keep it low so it sticks to the facts!
    )
    return response.choices[0].message.content


# --- 5. EXECUTE THE PIPELINE ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" PROJECT AEGIS - MANUAL RAG PIPELINE ")
    print("="*50 + "\n")
    
    # >>> CHANGE THIS QUESTION TO MATCH YOUR PAPER <<<
    user_query = "What is the main objective of this research paper?"
    
    # Execute the steps
    chunks, metadata = retrieve_context(user_query, top_k=3)
    system_msg, user_msg = build_prompt(user_query, chunks)
    answer = generate_answer(system_msg, user_msg)
    
    # Print the final result
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(answer)
    
    print("\n" + "="*50)
    print("SOURCES CITED:")
    print("="*50)
    for i, md in enumerate(metadata):
        print(f"- Source {i+1}: {md.get('source', 'Unknown')} (Page {md.get('page', 'Unknown')})")
