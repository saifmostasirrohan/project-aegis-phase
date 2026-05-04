import os
from dotenv import load_dotenv
from langchain_chroma import Chroma # <-- Updated import!
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. INITIALIZATION ---
load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", collection_name="aegis_papers", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)

# --- 2. THE HISTORY-AWARE RETRIEVER ---
# This prompt asks the LLM to rewrite the question based on chat history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


# --- 3. THE ANSWERING CHAIN ---
# This is our standard RAG prompt, but using MessagesPlaceholder for the history
qa_system_prompt = (
    "You are an expert AI research assistant for Project Aegis. "
    "Answer ONLY using the provided <context>. If the answer is not in the context, "
    "explicitly say 'I cannot find this information'.\n\n"
    "<context>\n{context}\n</context>"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


# --- 4. COMBINE INTO FINAL RAG CHAIN ---
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# --- 5. EXECUTE WITH MEMORY ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" PROJECT AEGIS - LANGCHAIN MEMORY RAG ")
    print("="*50 + "\n")
    
    # We will manually store the chat history in a list
    chat_history = []
    
    # Question 1
    q1 = "What is the main objective of this research paper?"
    print(f"User: {q1}")
    res1 = rag_chain.invoke({"input": q1, "chat_history": chat_history})
    print(f"AI: {res1['answer']}\n")
    
    # Add Q1 to history
    chat_history.extend([HumanMessage(content=q1), AIMessage(content=res1["answer"])])
    
    # Question 2 (Requires memory of Q1 to make sense!)
    q2 = "What specific clinical data did they use to achieve that objective?"
    print(f"User: {q2}")
    res2 = rag_chain.invoke({"input": q2, "chat_history": chat_history})
    print(f"AI: {res2['answer']}")
