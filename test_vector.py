from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings

# Use 5 dummy docs
docs = [
    Document(page_content="Value iteration is a method of computing optimal policies.", metadata={"source": "doc1"}),
    Document(page_content="Policy iteration involves policy evaluation and improvement.", metadata={"source": "doc2"}),
    Document(page_content="Reinforcement learning often uses Bellman equations.", metadata={"source": "doc3"}),
    Document(page_content="Q-learning is a model-free approach to RL.", metadata={"source": "doc4"}),
    Document(page_content="Markov decision processes model sequential decisions.", metadata={"source": "doc5"}),
]

# Extract text + metadata
texts = [doc.page_content for doc in docs]
metadatas = [doc.metadata for doc in docs]

# Use Ollama for consistency (can be slow, but only 5 texts)
embeddings = OllamaEmbeddings(model="llama3.2")
print("🔢 Embedding 5 dummy texts...")
embedded = embeddings.embed_documents(texts)

# Build vectorstore
print("🛠️ Building FAISS vectorstore (dummy)...")
vectorstore = FAISS.from_embeddings(
    [(text, embedding) for text, embedding in zip(texts, embedded)],
    metadatas=metadatas,
    embedding=embeddings
)

# Save for test compatibility
vectorstore.save_local("faiss_index_dummy")
print("✅ Dummy FAISS vectorstore saved to 'faiss_index_dummy'")
