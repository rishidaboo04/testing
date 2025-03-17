from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any, Optional
import asyncio

# Import existing RAG pipeline components
from main2 import create_document_store, create_rag_pipeline

# Add these imports to app.py
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Initialize FastAPI app
app = FastAPI(
    title="RAG LLM API",
    description="REST API for Question Answering using RAG Pipeline",
    version="1.0.0"
)

# Add these lines after creating the FastAPI app
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str

# Update the DocumentInfo Pydantic model to include more fields
class DocumentInfo(BaseModel):
    id: str
    content: str
    meta: Dict[str, Any] = {
        "file_name": str,
        "line_number": Optional[int],
        "url": Optional[str]
    }
    word_count: int
    content_preview: str

class RAGResponse(BaseModel):
    answer: str
    retrieved_documents: List[DocumentInfo]
    execution_time: float

# Initialize RAG components at startup
document_store = None
text_embedder = None
retriever = None
rag_pipeline = None

@app.on_event("startup")
async def startup_event():
    global document_store, text_embedder, retriever, rag_pipeline
    document_store = create_document_store()
    text_embedder, retriever, rag_pipeline = create_rag_pipeline(document_store)

# Update the ask_question endpoint
@app.post("/ask", response_model=RAGResponse)
async def ask_question(request: QuestionRequest):
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Embed the question
        text_embedding_result = text_embedder.run(text=request.question)
        
        # Retrieve documents
        retriever_result = retriever.run(query_embedding=text_embedding_result["embedding"])
        retrieved_docs = retriever_result["documents"]
        
        # Run the pipeline
        response = await rag_pipeline.run_async(
            {
                "text_embedder": {"text": request.question},
                "prompt_builder": {"question": request.question}
            }
        )
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Format retrieved documents with more detailed information
        retrieved_docs_info = [
            DocumentInfo(
                id=doc.id,
                content=doc.content,
                meta={
                    "file_name": doc.meta.get("file_name", "Unknown"),
                    "line_number": doc.meta.get("line_number", 1),  # Default to 1 if not present
                    "url": doc.meta.get("url", "")
                },
                word_count=len(doc.content.split()),
                content_preview=doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
            )
            for doc in retrieved_docs
        ]
        
        return RAGResponse(
            answer=response['llm']['replies'][0].text,
            retrieved_documents=retrieved_docs_info,
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "document_count": document_store.count_documents()}


# Add this route to serve the chat interface
@app.get("/chat", response_class=HTMLResponse)
async def chat_interface(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)