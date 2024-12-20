import os
import sys
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SimpleNodeParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
RAW_DATA_DIR = "./data/raw_transcripts"
PROCESSED_DATA_DIR = "./data/processed_index"
OLLAMA_MODEL = "nhat:latest"


def check_ollama_llm():
    try:
        llm = Ollama(model=OLLAMA_MODEL)
        response = llm.complete("Hello, are you working?")
        print(f"Ollama LLM check: {response}")
        return True
    except Exception as e:
        print(f"Error checking Ollama LLM: {e}")
        return False


def process_transcripts():
    if not check_ollama_llm():
        print(
            "Ollama LLM is not accessible. Please ensure Ollama is running and the model is available."
        )
        sys.exit(1)

    # Configure global settings
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.llm = Ollama(model=OLLAMA_MODEL)

    # Check if index already exists
    if os.path.exists(PROCESSED_DATA_DIR):
        print("Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=PROCESSED_DATA_DIR)
        index = load_index_from_storage(storage_context)

        # Check for new documents
        existing_docs = set(
            os.path.basename(doc.metadata["file_name"])
            for doc in index.docstore.docs.values()
        )
        all_docs = set(os.listdir(RAW_DATA_DIR))
        new_docs = all_docs - existing_docs

        if new_docs:
            print(f"Found {len(new_docs)} new documents. Updating index...")
            documents = SimpleDirectoryReader(
                RAW_DATA_DIR, filename_as_id=True
            ).load_data(filename_filter=lambda x: x in new_docs)
            parser = SimpleNodeParser.from_defaults()
            nodes = parser.get_nodes_from_documents(documents)
            index.insert_nodes(nodes)
        else:
            print("No new documents found. Index is up to date.")
    else:
        print("Creating new index...")
        documents = SimpleDirectoryReader(RAW_DATA_DIR, filename_as_id=True).load_data()
        index = VectorStoreIndex.from_documents(documents)

    # Persist the index
    index.storage_context.persist(persist_dir=PROCESSED_DATA_DIR)
    print(
        f"Index {'updated' if os.path.exists(PROCESSED_DATA_DIR) else 'created'} and saved to {PROCESSED_DATA_DIR}"
    )

    return index


def query_index(index, query_text):
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return response


if __name__ == "__main__":
    index = process_transcripts()

    # Example query
    query = "What is the main idea of Nietzsche's philosophy?"
    query = "What are key concepts in study of philosophy?"
    response = query_index(index, query)

    print("\nQuery Results:")
    print(response)
    print("\nSource Nodes:")
    for node in response.source_nodes:
        print(f"\n--- Source Node ---")
        print(f"Content: {node.node.text[:200]}...")
        print(f"Source: {node.node.metadata.get('file_name', 'Unknown')}")
