import os
from docling.document_converter import DocumentConverter
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Create a converter object for docling
converter = DocumentConverter()

def load_pdfs(path):
    documents = []
    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(path, filename)
            print(f"Loading {filepath}...")

            result = converter.convert(filepath)
            text = str(result.document)  # ✅ Safe fallback that returns readable text

            doc = Document(page_content=text, metadata={"source": filename})
            documents.append(doc)

    return documents

def export_markdown_files(documents, output_dir="data/processed_markdown"):
    os.makedirs(output_dir, exist_ok=True)
    for doc in documents:
        filename = doc.metadata.get("source", "unknown").replace(".pdf", ".md")
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            f.write(doc.page_content)
        print(f"Exported {filepath}")

def split_documents_by_structure(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

#this is not needed but helps see the character count in each pdf
def count_characters(documents):
    for doc in documents:
        print(f"{doc.metadata['source']}: {len(doc.page_content)} characters")




#call these functions in rag.py
