import chromadb
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. THE CLEANING FUNCTION ---
def clean_text(text):
    """Fixes broken hyphenations and scrambled two-column PDF text."""
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    return text

# --- 2. LOAD AND CLEAN ---
pdf_path = "paper3.pdf" 
print(f"1. Loading {pdf_path}...")
loader = PyPDFLoader(pdf_path)
pages = loader.load()

print("2. Cleaning extracted text...")
for page in pages:
    # We overwrite the messy PDF text with our cleaned text
    page.page_content = clean_text(page.page_content)

# --- 3. SPLIT THE TEXT ---
chunk_size = 500      
chunk_overlap = 50    
print(f"3. Splitting text (Size: {chunk_size}, Overlap: {chunk_overlap})...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
chunks = text_splitter.split_documents(pages)
print(f"   -> Paper divided into {len(chunks)} clean chunks.\n")

# --- 4. INGEST INTO CHROMA DATABASE ---
print("4. Initializing ChromaDB persistent client...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="aegis_papers")

documents = []
metadatas = []
ids = []

for i, chunk in enumerate(chunks):
    documents.append(chunk.page_content)
    
    # Force clean, explicit metadata
    clean_metadata = {
        "source": chunk.metadata.get("source", "unknown"),
        "page": chunk.metadata.get("page", 0),
        "chunk_id": i
    }
    metadatas.append(clean_metadata)
    ids.append(f"paper_{i}")

print(f"5. Adding {len(documents)} chunks to the 'aegis_papers' collection...")
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("\n--- PIPELINE COMPLETE ---")
print(f"Total chunks safely stored in 'aegis_papers': {collection.count()}")
