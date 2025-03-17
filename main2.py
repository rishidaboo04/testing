import os
import asyncio
import hashlib
import time
import sys

from haystack import Pipeline, Document, component, ComponentError
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever

from haystack.core.component import Component
from typing import List, Dict, Any




def create_document_store(folder_path: str = "text_files/") -> InMemoryDocumentStore:
    """
    Initialize an InMemoryDocumentStore using multiple text files in a folder.

    Args:
        folder_path (str): Path to the folder containing text files.

    Returns:
        InMemoryDocumentStore: Initialized document store.
    """
    document_store = InMemoryDocumentStore()

    if document_store.count_documents() == 0:
        print("Creating new document store and indexing documents from text files...")

        # Iterate over each text file in the specified folder
        file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
        
        docs = []

        # Read each file line by line
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            # Generate documents from each line in the file
            for idx, line in enumerate(lines):
                if line.strip():  # Skip empty lines
                    # Add 1 to idx to start line numbers from 1 instead of 0
                    line_number = idx + 1
                    docs.append(
                        Document(
                            content=line.strip(),  # The content is the whole line, stripped of leading/trailing whitespace
                            meta={"file_name": os.path.basename(file_path), "line_number": line_number},  # Optional metadata: file name and line number
                            id=hashlib.sha256(line.encode()).hexdigest()  # Unique ID for each document based on its content
                        )
                    )

        # Embed documents
        doc_embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        doc_embedder.warm_up()

        # Time document embedding
        start_time = time.time()
        docs_with_embeddings = doc_embedder.run(docs)
        embedding_time = time.time() - start_time
        print(f"Document embedding took {embedding_time:.4f} seconds")

        # Write documents to the store
        document_store.write_documents(docs_with_embeddings["documents"])
    else:
        print(f"Using existing document store with {document_store.count_documents()} documents")

    return document_store


def create_rag_pipeline(document_store: InMemoryDocumentStore) -> tuple:
    """
    Create a RAG Pipeline.

    Args:
        document_store (InMemoryDocumentStore): Initialized document store.

    Returns:
        tuple: Configured text embedder, retriever, and pipeline
    """
    # Initialize components
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    text_embedder.warm_up()

    # Change retriever to InMemoryEmbeddingRetriever
    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=5  # Reduces retrieved documents to 5
    )

    # Create a prompt template as a string instead of ChatMessage objects
    prompt_template = """
Given the following context information, provide a detailed and accurate answer to the question. If the context does not contain sufficient information to fully answer the question, clearly state what information is missing.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}

Instructions for answering:
- Base your answer primarily on the provided context
- Aim for a clear, concise, and informative response
- If no relevant information is found, say "I cannot find sufficient information to answer this question."

Answer:
"""

    # Use PromptBuilder instead of ChatPromptBuilder
    from haystack.components.builders import PromptBuilder
    prompt_builder = PromptBuilder(template=prompt_template)
    
    # Configure the LLM
    chat_generator = HuggingFaceLocalGenerator(
        model="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        generation_kwargs={
            "max_new_tokens": 100,
            "temperature": 0.9,
            "do_sample": True
        }
    )

    # Build the Pipeline (changed from AsyncPipeline)
    basic_rag_pipeline = Pipeline()
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", chat_generator)

    # Connect the components
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")

    return text_embedder, retriever, basic_rag_pipeline


def run_pipeline(pipeline: Pipeline, text_embedder, retriever, question: str):
    """
    Run the RAG pipeline synchronously and print results.

    Args:
        pipeline (Pipeline): Configured RAG pipeline.
        text_embedder: Text embedding component
        retriever: Document retriever component
        question (str): Question to be answered.
    """
    start_time = time.time()

    # Embed the text
    text_embedding_result = text_embedder.run(text=question)

    # Retrieve documents
    retriever_result = retriever.run(query_embedding=text_embedding_result["embedding"])
    retrieved_docs = retriever_result["documents"]

    # Run the full pipeline
    response = pipeline.run(
        {"text_embedder": {"text": question}, "prompt_builder": {"question": question}}
    )

    pipeline_time = time.time() - start_time

    # Extract LLM reply - adjust based on your LLM's output format
    llm_reply = response['llm']['generated_text']

    # Print the final output in a clear and readable format
    print("\n" + "=" * 80)
    print("Pipeline Execution Summary")
    print("=" * 80)
    print(f"Total Pipeline Execution Time: {pipeline_time:.4f} seconds")

    print("\n" + "=" * 80)
    print("Retrieved Documents")
    print("=" * 80)

    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\nDocument {i}:")
        print("-" * 40)
        if hasattr(doc, 'meta'):
            for key, value in doc.meta.items():
                print(f"{key.capitalize()}: {value}")
        print(f"Document ID: {doc.id[:7]}...")
        print(f"Content Preview ({len(doc.content.split())} words total):")
        print(doc.content[:300] + "...")
        print("-" * 40)

    print("\n" + "=" * 80)
    print("Final Response")
    print("=" * 80)
    print(llm_reply)
    print("=" * 80 + "\n")


def interactive_rag_pipeline():
    """
    Interactive RAG pipeline with continuous questioning.
    """
    # Create document store and pipeline components
    document_store = create_document_store(folder_path="text_files/")
    text_embedder, retriever, basic_rag_pipeline = create_rag_pipeline(document_store)

    print("Interactive RAG Pipeline")
    print("Type your questions. Enter 'exit' or press Ctrl+C to quit.")

    try:
        while True:
            # Prompt for input
            question = input("\nEnter your question (or 'exit' to quit): ").strip()

            # Check for exit condition
            if question.lower() in ['exit', 'quit', 'q']:
                print("\nExiting the RAG pipeline. Goodbye!")
                break

            # Skip empty inputs
            if not question:
                continue

            # Run the pipeline for the current question
            run_pipeline(basic_rag_pipeline, text_embedder, retriever, question)

    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting the RAG pipeline. Goodbye!")
        sys.exit(0)


def main():
    """
    Main function to run the interactive RAG pipeline.
    """
    interactive_rag_pipeline()


if __name__ == "__main__":
    main()
