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
from langchain.schema.runnable import RunnableMap
from operator import itemgetter 
from data_prep import load_pdfs, export_markdown_files, split_documents_by_structure, count_characters

load_dotenv()

# Choose model
#CLAUDE_KEY = os.getenv("CLAUDE_KEY")
#MODEL = "claude-3-7-sonnet-20250219"
MODEL = "llama3.2"

llm = OllamaLLM(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

parser = StrOutputParser()

if os.path.exists("split_docs.pkl"):
    print("📂 Loading preprocessed document chunks from split_docs.pkl...")
    with open("split_docs.pkl", "rb") as f:
        split_docs = pickle.load(f)
else:
    print("🛠️ Processing raw PDFs...")
    raw_docs = load_pdfs("data")
    export_markdown_files(raw_docs)
    count_characters(raw_docs)
    split_docs = split_documents_by_structure(raw_docs)

    with open("split_docs.pkl", "wb") as f:
        pickle.dump(split_docs, f)
    print("✅ Saved split_docs to split_docs.pkl")

print(f"\n✅ Total document chunks: {len(split_docs)}")
for i, chunk in enumerate(split_docs[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content[:300])
    print(f"Source: {chunk.metadata.get('source')}")

# Save processed docs
if not os.path.exists("split_docs.pkl"):
    with open("split_docs.pkl", "wb") as f:
        pickle.dump(split_docs, f)

# Load or build FAISS vectorstore
faiss_path = "faiss_index"
if os.path.exists(faiss_path):
    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    print("📂 Loaded FAISS vectorstore")
else:
    print("🔍 Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
    vectorstore.save_local(faiss_path)
    print("✅ FAISS vectorstore saved")

# Build hybrid retriever
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = 6
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# RAG prompt
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

prompt = PromptTemplate.from_template(template)

# Final RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = RunnableMap({
    "context": itemgetter("question") | hybrid_retriever | format_docs,
    "query": itemgetter("question")
}) | prompt | llm | parser

# Interactive Q&A
print("\n💬 Ask a question (type 'exit' to quit):")
while True:
    question = input("🧠 > ")
    if question.strip().lower() in ["exit", "quit"]:
        break

    print("\n🤖 Answer:\n")
    for chunk in rag_chain.stream({"question": question}):
        print(chunk, end="", flush=True)
    print("\n" + "-" * 60)
