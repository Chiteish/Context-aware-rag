import os
import json
import shutil
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

# RAG Pipeline modules
from src.load_documents import load_pdf
from src.chunk_text import split_documents
from src.embeddings import get_embeddings_model
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.rag_pipeline import build_rag_chain

# In-memory session store
# sessions = {
#     "session_id": {
#         "qa_chain": qa_chain_object or None,
#         "metadata": { ... }
#     }
# }
sessions = {}

def get_session_dir(session_id: str) -> str:
    # Basic directory traversal security check
    if not all(c.isalnum() or c == '-' for c in session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID format")
    return os.path.join("data", "sessions", session_id)

def get_or_create_session(session_id: str):
    global sessions
    
    # 1. Check in-memory registry
    if session_id in sessions:
        return sessions[session_id]
        
    # 2. Check if cached on disk
    session_dir = get_session_dir(session_id)
    config_path = os.path.join(session_dir, "metadata.json")
    index_path = os.path.join(session_dir, "faiss_index")
    
    session_metadata = {
        "document_name": "",
        "chunks_count": 0,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "mistral (Ollama)",
        "status": "awaiting_document"
    }
    
    qa_chain = None
    
    if os.path.exists(config_path) and os.path.exists(index_path):
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            doc_name = config_data.get("document_name", "")
            if doc_name:
                print(f"Loading cached RAG pipeline for session {session_id} ({doc_name})...")
                embedding_model = get_embeddings_model()
                from langchain_community.vectorstores import FAISS
                vector_store = FAISS.load_local(
                    index_path,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
                try:
                    chunks_count = vector_store.index.ntotal
                except AttributeError:
                    chunks_count = -1
                
                retriever = get_retriever(vector_store)
                qa_chain = build_rag_chain(retriever)
                
                session_metadata["document_name"] = doc_name
                session_metadata["chunks_count"] = chunks_count
                session_metadata["status"] = "ready"
                print(f"Session {session_id} loaded successfully!")
        except Exception as e:
            session_metadata["status"] = f"error: {str(e)}"
            print(f"Error loading session {session_id} from cache: {e}")
            
    sessions[session_id] = {
        "qa_chain": qa_chain,
        "metadata": session_metadata
    }
    return sessions[session_id]

def process_pdf(session_id: str, file_path: str, filename: str):
    global sessions
    session = get_or_create_session(session_id)
    session_metadata = session["metadata"]
    
    session_metadata["status"] = "processing"
    session_metadata["document_name"] = filename
    session_metadata["chunks_count"] = 0
    
    try:
        embedding_model = get_embeddings_model()
        documents = load_pdf(file_path)
        chunks = split_documents(documents)
        chunks_count = len(chunks)
        
        vector_store = create_vector_store(chunks, embedding_model)
        
        session_dir = get_session_dir(session_id)
        index_path = os.path.join(session_dir, "faiss_index")
        os.makedirs(index_path, exist_ok=True)
        vector_store.save_local(index_path)
        
        # Save session metadata
        config_path = os.path.join(session_dir, "metadata.json")
        with open(config_path, "w") as f:
            json.dump({"document_name": filename}, f)
            
        retriever = get_retriever(vector_store)
        session["qa_chain"] = build_rag_chain(retriever)
        
        session_metadata["chunks_count"] = chunks_count
        session_metadata["status"] = "ready"
        print(f"RAG pipeline initialized for session {session_id} with document: {filename}")
    except Exception as e:
        session_metadata["status"] = f"error: {str(e)}"
        print(f"Error processing PDF for session {session_id}: {e}")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("DocuRAG session-based API started.")
    yield
    # Shutdown

app = FastAPI(title="DocuRAG Assistant API", lifespan=lifespan)

# Enable CORS for external testing/debugging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/upload")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    session_dir = get_session_dir(session_id)
    os.makedirs(session_dir, exist_ok=True)
    file_path = os.path.join(session_dir, file.filename)
    
    try:
        # Save the uploaded file to the session directory
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF in a background thread to keep FastAPI responsive
        await run_in_threadpool(process_pdf, session_id, file_path, file.filename)
        
        return {"status": "success", "message": f"Successfully processed {file.filename}"}
    except Exception as e:
        # Clean up file on failure if it exists
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ClearRequest(BaseModel):
    session_id: str

class ResetRequest(BaseModel):
    session_id: str

class SourceDoc(BaseModel):
    page_content: str
    page: int
    source: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]

@app.get("/api/status")
async def get_status(session_id: str):
    session = get_or_create_session(session_id)
    return session["metadata"]

def is_broad_question(query):

        broad_keywords = [
        "important topics",
        "main topics",
        "all topics",
        "summarize",
        "summary",
        "overview",
        "entire document",
        "whole document",
        "this unit",
        "this chapter",
        "explain the unit",
        "key concepts"
    ]

        query = query.lower()

        return any(keyword in query for keyword in broad_keywords)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    global sessions
    session_id = request.session_id.strip()
    session = get_or_create_session(session_id)
    qa_chain = session["qa_chain"]
    session_metadata = session["metadata"]

    if qa_chain is None:
        raise HTTPException(
            status_code=503, 
            detail=f"RAG model is not ready for this session. Current status: {session_metadata['status']}"
        )
    
    query = request.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query string")
    
    try:
        from langchain_core.messages import HumanMessage, AIMessage

        def event_generator():
            # 1. Retrieve documents
            # 1. Detect question type
            if is_broad_question(query):
                k = 12
                print("Broad question detected")
            else:
                k = 3
                print("Specific question detected")

            # 2. Create retriever dynamically
            vector_store = qa_chain.retriever.vectorstore
            print("Total indexed chunks:", len(vector_store.index_to_docstore_id))

            dynamic_retriever = get_retriever(
                vector_store,
                k=k
            )

# 3. Retrieve relevant documents
            docs = dynamic_retriever.invoke(query)
            print("\n========== RETRIEVAL DEBUG ==========")
            print("User Question:", query)
            print("Retrieval Mode:", "Broad" if k == 12 else "Specific")
            print("Total Chunks Retrieved:", len(docs))

            for i, doc in enumerate(docs, start=1):
                print(f"\n--- CHUNK {i} ---")
                print("Page:", doc.metadata.get("page", "Unknown") + 1)
                print("Content:")
                print(doc.page_content)

            print("\n====================================\n")

            print(f"Retrieved {len(docs)} chunks")
            
            # Format sources to send
            sources = []
            for doc in docs:
                page = doc.metadata.get("page", 0) + 1
                source_file = os.path.basename(doc.metadata.get("source", session_metadata["document_name"]))
                sources.append({
                    "page_content": doc.page_content,
                    "page": page,
                    "source": source_file
                })
            
            # Yield sources as the first SSE message
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # 2. Re-create the prompt with history
            history_lines = []
            for m in qa_chain.memory.messages:
                if isinstance(m, HumanMessage):
                    history_lines.append(f"User: {m.content}")
                elif isinstance(m, AIMessage):
                    history_lines.append(f"AI: {m.content}")
            history_text = "\n".join(history_lines)

            context = "\n\n".join(d.page_content for d in docs)
            prompt = (
                (f"Conversation history:\n{history_text}\n\n" if history_text else "") +
                f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )

            # 3. Stream from the LLM
            full_answer = ""
            for chunk in qa_chain.llm.stream(prompt):
                # Extract string content from LLM response chunk
                if hasattr(chunk, "content"):
                    token = chunk.content
                else:
                    token = str(chunk)
                
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            # 4. Save to memory
            qa_chain.memory.add_user_message(query)
            qa_chain.memory.add_ai_message(full_answer)
            
            yield "data: {\"type\": \"done\"}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear")
async def clear_chat(request: ClearRequest):
    global sessions
    session = get_or_create_session(request.session_id)
    qa_chain = session["qa_chain"]
    if qa_chain is None or not hasattr(qa_chain, "memory"):
        raise HTTPException(status_code=503, detail="RAG model memory is not initialized for this session")
    
    try:
        qa_chain.memory.clear()
        return {"status": "success", "message": "Chat history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")

@app.post("/api/reset-session")
@app.post("/reset-session")
async def reset_session(request: ResetRequest):
    global sessions
    session_id = request.session_id.strip()
    
    # 1. Clean up in-memory registry
    if session_id in sessions:
        del sessions[session_id]
        
    # 2. Clean up files on disk
    session_dir = get_session_dir(session_id)
    if os.path.exists(session_dir):
        try:
            shutil.rmtree(session_dir)
            print(f"Session {session_id} assets deleted from disk.")
        except Exception as e:
            print(f"Error deleting session directory {session_dir}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete session files: {str(e)}")
            
    return {"status": "success", "message": f"Session {session_id} has been reset."}

# Ensure the static files directory exists and mount it
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
