import pickle
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM

load_dotenv()

# Load split_docs from disk
with open("split_docs.pkl", "rb") as f:
    split_docs = pickle.load(f)

# Choose model
MODEL = "llama3.2"
llm = OllamaLLM(model=MODEL)
chain = llm | StrOutputParser()

# Prompt template
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

# Simulate retrieval with a few chunks
sample_docs = split_docs[:5]
context = "\n\n".join(doc.page_content for doc in sample_docs)

inputs = {
    "context": context,
    "query": "What is the difference between value iteration and policy iteration?"
}

print("\n🧪 Running test prompt...")
response = chain.invoke(prompt.format(**inputs))
print("\n📤 Final Response:\n")
print(response)