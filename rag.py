import os
import pickle
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from docling.document_converter import DocumentConverter
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from operator import itemgetter 
from data_prep import load_pdfs, export_markdown_files, split_documents_by_structure, count_characters

load_dotenv()

'''USE THIS IF YOU DECIDE ON USING CLAUDE'''
#CLAUDE_KEY = os.getenv("CLAUDE_KEY")
#MODEL = "claude-3-7-sonnet-20250219"

'''USE THIS IF YOU DECIDE ON USING OLLAMA'''
MODEL = "llama3.2"

if MODEL.startswith("claude"):
    llm = ChatAnthropic(model=MODEL, api_key=CLAUDE_KEY)
else:
    llm = OllamaLLM(model=MODEL)
    embeddings = OllamaEmbeddings(model=MODEL)
    
    
#llm.invoke("tell me a joke")
parser = StrOutputParser()
chain = llm | parser
#chain.invoke("tell me a joke")

#call the functions completed in data_prep.py here 

print("📥 Loading PDFs...")
raw_docs = load_pdfs("data")  # or wherever your PDFs are

print("📝 Exporting markdown...")
export_markdown_files(raw_docs)

print("🔤 Character count...")
count_characters(raw_docs)

print("✂️ Splitting documents...")
split_docs = split_documents_by_structure(raw_docs)

# ✅ Quick verification test
print("\n✅ Split document verification:")
print(f"Total chunks: {len(split_docs)}")

# Print the first few chunks to preview their content
for i, chunk in enumerate(split_docs[:3]):  # Show first 3 chunks
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content[:300])  # Show first 300 characters
    print(f"Source: {chunk.metadata.get('source')}")

#question2 starts here
template = """
You are an AI assistant built to answer questions strictly using the information from retrieved documents.

### Retrieved Context:
<context>
{context}
</context>

### User Query:
<query>
{query}
</query>

### How to Answer:
1. Use only information found in the context above.
2. Do not use prior knowledge or external information — only what is in the context.
3. If multiple pieces of context conflict, acknowledge this in your response.
4. If there is not enough information to answer, say so explicitly.

### Response Format:
<relevant_sources>
Summarize the exact parts of the retrieved context that are useful for answering the query.
</relevant_sources>

<response>
Write a clear, direct answer using only the information from the relevant_sources.
If the query cannot be answered, explain that and why.
</response>

### Important Guidelines:
- No guessing or hallucinating.
- Do not provide examples or extra details that aren’t in the context.
- Be objective, accurate, and concise.
- Quote and cite the source if necessary.
"""

# Create a prompt template from your string
prompt = PromptTemplate.from_template(template)

# Choose a small sample of split_docs to simulate a retrieval
sample_docs = split_docs[:5]  # pretend these are retrieved docs

# Combine all their content into one string (simulate `context`)
context = "\n\n".join(doc.page_content for doc in sample_docs)

# Create an input dictionary
inputs = {
    "context": context,
    "query": "What is the difference between value iteration and policy iteration?"
}

# Run the chain
print("\n🧪 Running test prompt...")
response = chain.invoke(prompt.format(**inputs))
print("\n📤 Final Response:\n")
print(response)

# Save the split_docs list to a file so we don't repeat all the work next time
if not os.path.exists("split_docs.pkl"):
    with open("split_docs.pkl", "wb") as f:
        pickle.dump(split_docs, f)
    print("🗃️ Saved split_docs to split_docs.pkl")
else:
    print("🗃️ split_docs.pkl already exists — skipping save")

# ⚠️ Only build vectorstore if it doesn't already exist
faiss_path = "faiss_index"
if not os.path.exists(faiss_path):
    print("🔍 Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
    vectorstore.save_local(faiss_path)
    print("✅ FAISS vectorstore saved to 'faiss_index'")
else:
    print("📂 FAISS vectorstore already exists — skipping creation")

# Load it
vectorstore = FAISS.load_local(faiss_path, embeddings)
retriever_dense = vectorstore.as_retriever()

# Create sparse retriever (BM25)
retriever_sparse = BM25Retriever.from_documents(split_docs)
retriever_sparse.k = 4

# Combine both using EnsembleRetriever
retriever = EnsembleRetriever(
    retrievers=[retriever_dense, retriever_sparse],
    weights=[0.5, 0.5]
)