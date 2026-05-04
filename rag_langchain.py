import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. INITIALIZATION ---
load_dotenv()
print("Waking up LangChain...")

# Connect to the exact same Chroma database from CP-03/CP-04
# We explicitly load the default sentence-transformer model so it can read your existing math
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", collection_name="aegis_papers", embedding_function=embeddings)

# Create a retriever that automatically fetches the top 3 chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize Groq
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)


# --- 2. THE PROMPT TEMPLATE ---
# Notice how clean this is compared to manually concatenating strings
template = """You are an expert AI research assistant for Project Aegis.
Answer ONLY using the provided <context>. If the answer is not in the context, explicitly say 'I cannot find this information'.

<context>
{context}
</context>

<question>
{question}
</question>"""

prompt = ChatPromptTemplate.from_template(template)


# --- 3. THE LCEL CHAIN ---
# Helper function to combine retrieved document chunks into one string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# This is LangChain Expression Language (LCEL)
# It pipes data left-to-right: Retriever -> Prompt -> LLM -> String Output
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 4. EXECUTE ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" PROJECT AEGIS - LANGCHAIN LCEL RAG ")
    print("="*50 + "\n")
    
    # Change this to a question you know the answer to!
    user_query = "What is the main objective of this research paper?"
    
    print(f"Question: {user_query}\n")
    print("Thinking...\n")
    
    # .invoke() triggers the entire pipeline automatically
    answer = rag_chain.invoke(user_query)
    
    print(answer)
