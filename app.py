
import os
import json
import shutil
import uvicorn
from dotenv import load_dotenv

load_dotenv()

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
sessions = {}


def get_session_dir(session_id: str) -> str:
    """Return the directory for a session."""

    if not all(c.isalnum() or c == '-' for c in session_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid session ID format"
        )

    return os.path.join("data", "sessions", session_id)


def get_or_create_session(session_id: str):
    """Get an existing session or create/load one."""

    global sessions

    if session_id in sessions:
        return sessions[session_id]

    session_dir = get_session_dir(session_id)
    config_path = os.path.join(session_dir, "metadata.json")
    index_path = os.path.join(session_dir, "faiss_index")

    session_metadata = {
        "document_name": "",
        "chunks_count": 0,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "openai/gpt-oss-120b (Groq)",
        "status": "awaiting_document"
    }

    qa_chain = None

    # Load cached FAISS index if available
    if os.path.exists(config_path) and os.path.exists(index_path):

        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)

            doc_name = config_data.get("document_name", "")

            if doc_name:
                print(
                    f"Loading cached RAG pipeline for session "
                    f"{session_id} ({doc_name})..."
                )

                embedding_model = get_embeddings_model()

                from langchain_community.vectorstores import FAISS

                vector_store = FAISS.load_local(
                    index_path,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )

                chunks_count = vector_store.index.ntotal

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
    """Process uploaded PDF and create its FAISS index."""

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

        vector_store = create_vector_store(
            chunks,
            embedding_model
        )

        session_dir = get_session_dir(session_id)
        index_path = os.path.join(session_dir, "faiss_index")

        os.makedirs(index_path, exist_ok=True)

        vector_store.save_local(index_path)

        # Save session metadata
        config_path = os.path.join(session_dir, "metadata.json")

        with open(config_path, "w") as f:
            json.dump({
                "document_name": filename
            }, f)

        retriever = get_retriever(vector_store)

        session["qa_chain"] = build_rag_chain(retriever)

        session_metadata["chunks_count"] = chunks_count
        session_metadata["status"] = "ready"

        print(
            f"RAG pipeline initialized for session {session_id} "
            f"with document: {filename}"
        )

    except Exception as e:
        session_metadata["status"] = f"error: {str(e)}"

        print(
            f"Error processing PDF for session {session_id}: {e}"
        )

        raise e


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("DocuRAG session-based API started.")
    yield


app = FastAPI(
    title="DocuRAG Assistant API",
    lifespan=lifespan
)


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# PDF UPLOAD ENDPOINT
# --------------------------------------------------

@app.post("/api/upload")
async def upload_document(
    session_id: str,
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    session_dir = get_session_dir(session_id)
    os.makedirs(session_dir, exist_ok=True)

    file_path = os.path.join(session_dir, file.filename)

    try:
        # Save uploaded PDF
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process PDF in background thread
        await run_in_threadpool(
            process_pdf,
            session_id,
            file_path,
            file.filename
        )

        return {
            "status": "success",
            "message": f"Successfully processed {file.filename}"
        }

    except Exception as e:

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

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


# --------------------------------------------------
# STATUS ENDPOINT
# --------------------------------------------------

@app.get("/api/status")
async def get_status(session_id: str):

    session = get_or_create_session(session_id)

    return session["metadata"]


# --------------------------------------------------
# BROAD QUESTION DETECTION
# --------------------------------------------------

def is_broad_question(query: str) -> bool:

    broad_keywords = [
        "important topics",
        "main topics",
        "major topics",
        "major topic",
        "all topics",
        "all major topics",
        "summarize",
        "summary",
        "overview",
        "entire document",
        "whole document",
        "this document",
        "covered in this document",
        "explain the document",
        "explain everything",
        "this unit",
        "this chapter",
        "explain the unit",
        "key concepts",
        "key points",
        "all chapters",
        "main concepts"
    ]

    query = query.lower().strip()

    return any(
        keyword in query
        for keyword in broad_keywords
    )


# --------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------

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
            detail=(
                "RAG model is not ready for this session. "
                f"Current status: {session_metadata['status']}"
            )
        )

    query = request.message.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Empty query string"
        )

    try:

        from langchain_core.messages import HumanMessage, AIMessage

        def event_generator():

            # ------------------------------------------
            # 1. GET VECTOR STORE
            # ------------------------------------------

            vector_store = qa_chain.retriever.vectorstore

            total_chunks = vector_store.index.ntotal

            # ------------------------------------------
            # 2. DETECT QUESTION TYPE
            # ------------------------------------------

            if is_broad_question(query):

                k = 20
                retrieval_mode = "Broad"

                print("Broad question detected")

            else:

                k = 3
                retrieval_mode = "Specific"

                print("Specific question detected")

            # Do not request more chunks than exist
            k = min(k, total_chunks)

            # ------------------------------------------
            # 3. RETRIEVAL DEBUG INFORMATION
            # ------------------------------------------

            print("\n========== RETRIEVAL DEBUG ==========")

            print("USER QUERY:", query)
            print("RETRIEVAL MODE:", retrieval_mode)
            print("RETRIEVAL K:", k)
            print("TOTAL DOCUMENT CHUNKS:", total_chunks)

            # ------------------------------------------
            # 4. CREATE DYNAMIC RETRIEVER
            # ------------------------------------------

            dynamic_retriever = get_retriever(
                vector_store,
                k=k
            )

            # ------------------------------------------
            # 5. RETRIEVE DOCUMENTS
            # ------------------------------------------

            docs = dynamic_retriever.invoke(query)

            print("TOTAL CHUNKS RETRIEVED:", len(docs))

            for i, doc in enumerate(docs, start=1):

                page_number = doc.metadata.get("page", 0)

                print(f"\n--- CHUNK {i} ---")
                print("Page:", page_number + 1)
                print("Content:")
                print(doc.page_content[:500])

            print("\n====================================\n")

            # ------------------------------------------
            # 6. FORMAT SOURCES
            # ------------------------------------------

            sources = []

            for doc in docs:

                page = doc.metadata.get("page", 0)

                source_file = os.path.basename(
                    doc.metadata.get(
                        "source",
                        session_metadata["document_name"]
                    )
                )

                sources.append({
                    "page_content": doc.page_content,
                    "page": page + 1,
                    "source": source_file
                })

            # Send sources through SSE
            yield (
                f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            )

            # ------------------------------------------
            # 7. BUILD CONVERSATION HISTORY
            # ------------------------------------------

            history_lines = []

            for m in qa_chain.memory.messages:

                if isinstance(m, HumanMessage):
                    history_lines.append(f"User: {m.content}")

                elif isinstance(m, AIMessage):
                    history_lines.append(f"AI: {m.content}")

            history_text = "\n".join(history_lines)

            if not history_text:
                history_text = "No previous conversation."

            # ------------------------------------------
            # 8. BUILD DOCUMENT CONTEXT
            # ------------------------------------------

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            # ------------------------------------------
            # 9. CREATE PROMPT
            # ------------------------------------------

            prompt = f"""
You are a document-based question-answering assistant.

Your task is to answer the user's question using the uploaded
document context.

STRICT RULES:

1. Answer using only information supported by the document context.
2. Do not invent facts or add information from outside the document.
3. Do not repeat a previous answer without checking the current question.
4. Answer the current question directly.
5. Use clear and simple explanations.
6. Organize broad answers using headings and bullet points.
7. For broad questions, cover all major topics that are actually
   represented in the supplied context.
8. For specific questions, focus only on the relevant information.
9. If the answer is not available in the context, say:
   "I couldn't find this information in the uploaded document."

CONVERSATION HISTORY:
{history_text}

IMPORTANT:
Conversation history is only for understanding follow-up questions.
Do not use it as a source of factual information.

UPLOADED DOCUMENT CONTEXT:
{context}

CURRENT USER QUESTION:
{query}

ANSWER:
"""

            # ------------------------------------------
            # 10. STREAM RESPONSE FROM GROQ
            # ------------------------------------------

            full_answer = ""

            for chunk in qa_chain.llm.stream(prompt):

                if hasattr(chunk, "content"):
                    token = chunk.content
                else:
                    token = str(chunk)

                # Ignore empty or non-text chunks
                if not token or not isinstance(token, str):
                    continue

                full_answer += token

                # Send token through SSE
                yield (
                    f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                )

            # ------------------------------------------
            # 11. SAVE CONVERSATION MEMORY
            # ------------------------------------------

            if full_answer.strip():

                qa_chain.memory.add_user_message(query)

                qa_chain.memory.add_ai_message(full_answer)

            # ------------------------------------------
            # 12. FINISH STREAM
            # ------------------------------------------

            yield (
                f"data: {json.dumps({'type': 'done'})}\n\n"
            )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )

    except Exception as e:

        print(f"Error in chat endpoint: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# --------------------------------------------------
# CLEAR CHAT HISTORY
# --------------------------------------------------

@app.post("/api/clear")
async def clear_chat(request: ClearRequest):

    global sessions

    session = get_or_create_session(request.session_id)

    qa_chain = session["qa_chain"]

    if qa_chain is None or not hasattr(qa_chain, "memory"):

        raise HTTPException(
            status_code=503,
            detail="RAG model memory is not initialized for this session"
        )

    try:

        qa_chain.memory.clear()

        return {
            "status": "success",
            "message": "Chat history cleared"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear memory: {str(e)}"
        )


# --------------------------------------------------
# RESET SESSION
# --------------------------------------------------

@app.post("/api/reset-session")
@app.post("/reset-session")
async def reset_session(request: ResetRequest):

    global sessions

    session_id = request.session_id.strip()

    # Remove from in-memory registry
    if session_id in sessions:
        del sessions[session_id]

    # Remove session files from disk
    session_dir = get_session_dir(session_id)

    if os.path.exists(session_dir):

        try:

            shutil.rmtree(session_dir)

            print(
                f"Session {session_id} assets deleted from disk."
            )

        except Exception as e:

            print(
                f"Error deleting session directory {session_dir}: {e}"
            )

            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete session files: {str(e)}"
            )

    return {
        "status": "success",
        "message": f"Session {session_id} has been reset."
    }


# --------------------------------------------------
# STATIC FILES
# --------------------------------------------------

os.makedirs("static", exist_ok=True)

app.mount(
    "/",
    StaticFiles(directory="static", html=True),
    name="static"
)


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
