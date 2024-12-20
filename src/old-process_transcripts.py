import os
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


def load_transcripts(directory):
    transcripts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                content = f.read()
                transcripts.append(
                    Document(page_content=content, metadata={"source": filename})
                )
    return transcripts


def process_transcripts():
    print("Processing transcripts...")

    raw_dir = "./data/raw_transcripts"
    processed_dir = "./data/processed_transcripts"
    os.makedirs(processed_dir, exist_ok=True)

    # Load raw transcripts
    raw_docs = load_transcripts(raw_dir)
    print(f"Loaded {len(raw_docs)} raw transcripts.")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(raw_docs)
    print(f"Split into {len(split_docs)} chunks.")

    # Create embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        [doc.page_content for doc in split_docs], show_progress_bar=True
    )

    # Save embeddings and metadata
    np.save(os.path.join(processed_dir, "embeddings.npy"), embeddings)

    metadata = [
        {"content": doc.page_content, "metadata": doc.metadata} for doc in split_docs
    ]
    np.save(os.path.join(processed_dir, "metadata.npy"), metadata)

    print(f"Created and saved embeddings and metadata in {processed_dir}")


if __name__ == "__main__":
    process_transcripts()
