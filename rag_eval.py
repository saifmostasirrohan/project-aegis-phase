import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Import your LangChain pipeline from CP-05
# (Ensure rag_langchain.py has rag_chain defined and accessible)
from rag_langchain import rag_chain 

load_dotenv()
print("1. Waking up RAGAS Evaluator...")

# --- 1. SETUP THE LLM JUDGE ---
# We configure RAGAS to use Groq as the grader instead of paying for OpenAI
judge_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
judge_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

ragas_llm = LangchainLLMWrapper(judge_llm)
ragas_emb = LangchainEmbeddingsWrapper(judge_embeddings)

# --- 2. THE GOLDEN DATASET ---
# These are the test questions and the factual answers we EXPECT to see.
questions = [
    "What is the main objective of this research paper?",
    "What dataset did Sui et al. (2025) use?"
]

ground_truths = [
    "The main objective is to evaluate and compare machine learning models for early diagnosis of HIV cases.",
    "Sui et al. (2025) used Clinical notes with 22->6 features."
]

# --- 3. GENERATE ANSWERS USING YOUR PIPELINE ---
print("2. Running test questions through your RAG pipeline...")
answers = []
contexts = []

for q in questions:
    # Run the query through your LCEL chain from CP-05
    response = rag_chain.invoke(q)
    
    # Extract the generated answer
    answers.append(response)
    
    # To grade 'Faithfulness', RAGAS needs to see the exact chunks that were retrieved.
    # Because our LCEL chain in rag_langchain.py returns a string, we will do a quick 
    # manual retrieval here just to feed the evaluator the context.
    from rag_langchain import retriever
    docs = retriever.invoke(q)
    contexts.append([doc.page_content for doc in docs])


# --- 4. FORMAT FOR RAGAS ---
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data)

# --- 5. RUN EVALUATION ---
print("3. Grading the results (This takes a few seconds)...")

# We are testing Faithfulness (No Hallucinations) and Relevancy (On-Topic)
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=ragas_llm,
    embeddings=ragas_emb
)

print("\n" + "="*50)
print(" RAGAS EVALUATION REPORT ")
print("="*50)
print(result)
