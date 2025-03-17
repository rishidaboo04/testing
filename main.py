import asyncio
import hashlib
import time
import sys

from datasets import load_dataset
from haystack import AsyncPipeline, Document
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore




def create_document_store(persist_path: str = "chroma_db") -> ChromaDocumentStore:
    """
    Initialize or load a ChromaDocumentStore.

    Args:
        persist_path (str): Path to persist the document store.

    Returns:
        ChromaDocumentStore: Initialized document store.
    """
    document_store = ChromaDocumentStore(persist_path=persist_path)

    if document_store.count_documents() == 0:
        print("Creating new document store and indexing documents...")
        dataset = load_dataset("bilgeyucel/seven-wonders", split="train")

        # Generate documents with stable IDs
        docs = [
            Document(
                content=doc["content"],
                meta=doc["meta"],
                id=hashlib.sha256(doc["content"].encode()).hexdigest()
            )
            for doc in dataset
        ]

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


def create_rag_pipeline(document_store: ChromaDocumentStore) -> tuple:
    """
    Create an Async RAG Pipeline.

    Args:
        document_store (ChromaDocumentStore): Initialized document store.

    Returns:
        tuple: Configured text embedder, retriever, and pipeline
    """
    # Initialize components
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    text_embedder.warm_up()

    retriever = ChromaEmbeddingRetriever(
        document_store=document_store, 
        top_k=5  # Reduces retrieved documents to 5
    )

    template = [
        ChatMessage.from_user(
            """
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
        )
    ]

    prompt_builder = ChatPromptBuilder(template=template)
    chat_generator = OpenAIChatGenerator(
        api_key=Secret.from_env_var("GROQ_API_KEY"),
        api_base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        generation_kwargs={"max_tokens": 512}
    )

    # Build the AsyncPipeline
    basic_rag_pipeline = AsyncPipeline()
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", chat_generator)

    # Connect components
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

    return text_embedder, retriever, basic_rag_pipeline


async def run_pipeline(pipeline: AsyncPipeline, text_embedder, retriever, question: str):
    """
    Run the RAG pipeline asynchronously and print results.

    Args:
        pipeline (AsyncPipeline): Configured RAG pipeline.
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
    response = await pipeline.run_async(
        {"text_embedder": {"text": question}, "prompt_builder": {"question": question}}
    )
    
    pipeline_time = time.time() - start_time

    # Extract LLM reply
    llm_reply = response['llm']['replies'][0].text

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


async def interactive_rag_pipeline():
    """
    Interactive RAG pipeline with continuous questioning.
    """
    # Create document store and pipeline components
    document_store = create_document_store()
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
            await run_pipeline(basic_rag_pipeline, text_embedder, retriever, question)

    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting the RAG pipeline. Goodbye!")
        sys.exit(0)


async def main():
    """
    Main async function to run the interactive RAG pipeline.
    """
    await interactive_rag_pipeline()


if __name__ == "__main__":
    asyncio.run(main())