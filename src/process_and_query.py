# import os
# import sys
# from llama_index.core import (
#     VectorStoreIndex,
#     SimpleDirectoryReader,
#     StorageContext,
#     load_index_from_storage,
#     Settings,
# )
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.ollama import Ollama
# from llama_index.core.node_parser import SimpleNodeParser
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Constants
# RAW_DATA_DIR = "./data/raw_transcripts"
# PROCESSED_DATA_DIR = "./data/processed_index"
# OLLAMA_MODEL = "nhat:latest"

# def check_ollama_llm():
#     try:
#         llm = Ollama(model=OLLAMA_MODEL)
#         response = llm.complete("Hello, are you working?")
#         print(f"Ollama LLM check: {response}")
#         return True
#     except Exception as e:
#         print(f"Error checking Ollama LLM: {e}")
#         return False

# def process_transcripts():
#     if not check_ollama_llm():
#         print("Ollama LLM is not accessible. Please ensure Ollama is running and the model is available.")
#         sys.exit(1)

#     # Configure global settings
#     Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     Settings.llm = Ollama(model=OLLAMA_MODEL)

#     # Check if index already exists
#     if os.path.exists(PROCESSED_DATA_DIR):
#         print("Loading existing index...")
#         storage_context = StorageContext.from_defaults(persist_dir=PROCESSED_DATA_DIR)
#         index = load_index_from_storage(storage_context)

#         # Check for new documents
#         existing_docs = set(os.path.basename(doc.metadata["file_name"]) for doc in index.docstore.docs.values())
#         all_docs = set(os.listdir(RAW_DATA_DIR))
#         new_docs = all_docs - existing_docs

#         if new_docs:
#             print(f"Found {len(new_docs)} new documents. Updating index...")
#             documents = SimpleDirectoryReader(RAW_DATA_DIR, filename_as_id=True).load_data(filename_filter=lambda x: x in new_docs)
#             parser = SimpleNodeParser.from_defaults()
#             nodes = parser.get_nodes_from_documents(documents)
#             index.insert_nodes(nodes)
#         else:
#             print("No new documents found. Index is up to date.")
#     else:
#         print("Creating new index...")
#         documents = SimpleDirectoryReader(RAW_DATA_DIR, filename_as_id=True).load_data()
#         index = VectorStoreIndex.from_documents(documents)

#     # Persist the index
#     index.storage_context.persist(persist_dir=PROCESSED_DATA_DIR)
#     print(f"Index {'updated' if os.path.exists(PROCESSED_DATA_DIR) else 'created'} and saved to {PROCESSED_DATA_DIR}")
#     return index

# def query_index(index, query_text):
#     query_engine = index.as_query_engine()
#     response = query_engine.query(query_text)

#     # Format the response
#     formatted_response = f"""Detailed Answer:
# {response.response}

# TLDR:
# [A brief summary of the main points]

# Suggested Follow-up Questions:
# 1. [First follow-up question]
# 2. [Second follow-up question]
# 3. [Third follow-up question]

# Source References:
# """
#     for i, node in enumerate(response.source_nodes):
#         formatted_response += f"[{i+1}] {node.node.metadata.get('file_name', 'Unknown')}\n"

#     return formatted_response, response.source_nodes

# def chat_loop(index):
#     print("Welcome to PhiloBot! Type 'exit' to end the conversation.")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             break

#         response, source_nodes = query_index(index, user_input)
#         print("\nPhiloBot:")
#         print(response)

#         print("\nSource Nodes:")
#         for node in source_nodes:
#             print(f"\n--- Source Node ---")
#             print(f"Content: {node.node.text[:200]}...")
#             print(f"Source: {node.node.metadata.get('file_name', 'Unknown')}")

#         print("\n" + "-"*50 + "\n")

# if __name__ == "__main__":
#     index = process_transcripts()
#     chat_loop(index)


import os
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
LLM_TIMEOUT = 60.0


def process_transcripts():
    print("Initializing embedding model and LLM...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=LLM_TIMEOUT)

    if os.path.exists(PROCESSED_DATA_DIR):
        print(f"Loading existing index from {PROCESSED_DATA_DIR}...")
        storage_context = StorageContext.from_defaults(persist_dir=PROCESSED_DATA_DIR)
        index = load_index_from_storage(storage_context)

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
        print(f"Creating new index from {RAW_DATA_DIR}...")
        documents = SimpleDirectoryReader(RAW_DATA_DIR, filename_as_id=True).load_data()
        index = VectorStoreIndex.from_documents(documents)

    index.storage_context.persist(persist_dir=PROCESSED_DATA_DIR)
    print(
        f"Index {'updated' if os.path.exists(PROCESSED_DATA_DIR) else 'created'} and saved to {PROCESSED_DATA_DIR}"
    )
    return index


def query_index(index, query_text):
    query_engine = index.as_query_engine()

    # First query for the main answer
    main_response = query_engine.query(query_text)

    # Second query for TLDR and follow-up questions
    followup_query = f"""Based on the following answer to the question "{query_text}", please provide:
1. A brief TLDR summary (2-3 sentences)
2. Three suggested follow-up questions

Answer:
{main_response.response}

Format your response exactly as follows:
TLDR: [Your TLDR here]

Suggested Follow-up Questions:
1. [First follow-up question]
2. [Second follow-up question]
3. [Third follow-up question]
"""

    followup_response = query_engine.query(followup_query)

    # Combine the responses
    formatted_response = f"""Detailed Answer:
{main_response.response}

{followup_response.response}
"""

    # Combine source nodes from both queries
    all_source_nodes = main_response.source_nodes + followup_response.source_nodes
    # Remove duplicates while preserving order
    unique_source_nodes = []
    seen = set()
    for node in all_source_nodes:
        if node.node.node_id not in seen:
            seen.add(node.node.node_id)
            unique_source_nodes.append(node)

    return formatted_response, unique_source_nodes


def chat_loop(index):
    print("Welcome to PhiloBot! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response, source_nodes = query_index(index, user_input)
        print("\nPhiloBot:")
        print(response)

        print("\nSource Nodes:")
        for i, node in enumerate(source_nodes, 1):
            print(f"\n--- Source Node {i} ---")
            print(f"Content: {node.node.text[:200]}...")
            print(f"Source: {node.node.metadata.get('file_name', 'Unknown')}")

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    index = process_transcripts()
    print(f"Vector store type: {type(index.vector_store)}")
    print(f"Number of documents in the index: {len(index.docstore.docs)}")

    if index.docstore.docs:
        sample_id = next(iter(index.docstore.docs))
        sample_doc = index.docstore.docs[sample_id]
        print(f"Sample document metadata: {sample_doc.metadata}")
        print(f"Sample document text (first 100 chars): {sample_doc.text[:100]}")
    else:
        print("Warning: No documents found in the index.")

    chat_loop(index)
