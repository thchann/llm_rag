import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from operator import itemgetter
from rag import (
    bm25_retriever,
    vector_retriever,
    EnsembleRetriever,
    prompt,
    llm,
    RunnableMap,
    parser
)
from tabulate import tabulate

def format_docs(docs):
    def clean(text):
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            if "orig=" in line:
                start = line.find("orig='")
                if start != -1:
                    end = line.find("'", start + 6)
                    if end != -1:
                        value = line[start + 6:end]
                        cleaned_lines.append(value)
                    else:
                        cleaned_lines.append(line)  
                else:
                    cleaned_lines.append(line)
            elif not any(x in line for x in ["ListItem", "RefItem", "DocItemLabel", "ContentLayer"]):
                cleaned_lines.append(line.strip())

        return "\n".join(cleaned_lines)

    return "\n\n".join(clean(doc.page_content) for doc in docs)

test_questions = {
    "What does PEAS stand for?": "Performance measure, Environment, Actuators, Sensors",
    "What is reinforcement learning?": "Learning by interacting with the environment using rewards and punishments.",
    "Compare value iteration and policy iteration.": "Both are dynamic programming algorithms; value iteration updates values directly, while policy iteration alternates between policy evaluation and improvement."
}

results = []

print("\nRunning RAG benchmark...\n")

for k in [4, 6, 8]:
    bm25_retriever.k = k
    vector_retriever.search_kwargs = {"k": k}

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    rag_chain = RunnableMap({
        "context": itemgetter("question") | hybrid_retriever | format_docs,
        "query": itemgetter("question")
    }) | prompt | llm | parser

    for question, expected in test_questions.items():
        start = time.time()
        answer = rag_chain.invoke({"question": question})
        latency = time.time() - start

        correct = "yes" if expected.lower()[:15] in answer.lower() else "no"

        results.append({
            "k": k,
            "question": question,
            "latency": round(latency, 2),
            "correct": correct,
            "answer": answer.strip().split("\n")[0][:50] + "..." 
        })

print(tabulate(results, headers="keys", tablefmt="grid"))
