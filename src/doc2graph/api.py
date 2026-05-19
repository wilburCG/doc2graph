"""FastAPI server for Doc2Graph.

Provides REST API endpoints for document processing, graph queries, and export.

Usage:
    uvicorn doc2graph.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from doc2graph.config import Config
from doc2graph.extractor.llm_extractor import LlmExtractor
from doc2graph.graph.importer import GraphImporter
from doc2graph.graph.merger import GraphMerger
from doc2graph.graph.neo4j_client import Neo4jClient
from doc2graph.graph.query import GraphQuery
from doc2graph.models.document import Document
from doc2graph.parser.pdf_parser import PdfParser
from doc2graph.parser.md_parser import MdParser
from doc2graph.parser.txt_parser import TxtParser
from doc2graph.parser.docx_parser import DocxParser

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

PARSERS = [PdfParser, MdParser, TxtParser, DocxParser]
config: Config | None = None
neo4j_client: Neo4jClient | None = None
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Progress Tracking (file-based, shared across workers)
# ---------------------------------------------------------------------------

_PROGRESS_DIR = Path("/tmp/doc2graph_tasks")
_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
_progress_lock = threading.Lock()


def _task_file(task_id: str) -> Path:
    return _PROGRESS_DIR / f"{task_id}.json"


def _set_progress(task_id: str, **kwargs):
    """Update progress for a task (file-based, works across workers)."""
    with _progress_lock:
        fpath = _task_file(task_id)
        data = {}
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text())
            except Exception:
                pass
        data.update(kwargs)
        data["updated_at"] = time.time()
        fpath.write_text(json.dumps(data))


def _get_progress(task_id: str) -> dict | None:
    fpath = _task_file(task_id)
    if not fpath.exists():
        return None
    try:
        return json.loads(fpath.read_text())
    except Exception:
        return None


def _cleanup_old_tasks(max_age: float = 300):
    """Remove task files older than max_age seconds."""
    now = time.time()
    with _progress_lock:
        for fpath in _PROGRESS_DIR.glob("*.json"):
            try:
                data = json.loads(fpath.read_text())
                if now - data.get("updated_at", 0) > max_age:
                    fpath.unlink()
            except Exception:
                pass


def _get_parser(file_path: Path):
    for parser_cls in PARSERS:
        if parser_cls.supports(file_path):
            return parser_cls()
    return None


def _get_client() -> Neo4jClient:
    global neo4j_client, config
    if neo4j_client is None:
        if config is None:
            config = Config.load()
        neo4j_client = Neo4jClient(
            config.neo4j.uri,
            config.neo4j.username,
            config.neo4j.password,
            config.neo4j.database,
        )
        neo4j_client.connect()
    return neo4j_client


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize connections on startup, close on shutdown."""
    global config
    config = Config.load()
    logger.info("Doc2Graph API started")
    yield
    global neo4j_client
    if neo4j_client:
        neo4j_client.close()
    logger.info("Doc2Graph API stopped")


app = FastAPI(
    title="Doc2Graph API",
    description="Convert documents to Neo4j knowledge graphs using LLMs",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ProcessResponse(BaseModel):
    entities: int
    relations: int
    source_doc: str


class MergeResponse(BaseModel):
    merged: int
    total_nodes: int
    total_relationships: int


class QueryRequest(BaseModel):
    cypher: str


class QueryResponse(BaseModel):
    results: list[dict]
    count: int


class StatsResponse(BaseModel):
    nodes: int
    relationships: int


class EntityItem(BaseModel):
    name: str
    type: str
    source: str | None = None


class RelationItem(BaseModel):
    source: str
    type: str
    target: str


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

# Mount the frontend static files
# __file__ = .../src/doc2graph/api.py → go up 3 levels to project root
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
if not FRONTEND_DIR.exists():
    # Fallback: search for frontend in common locations
    for candidate in [Path.cwd() / "frontend", Path(__file__).parents[2] / "frontend"]:
        if candidate.exists():
            FRONTEND_DIR = candidate
            break


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Frontend not found</h1>")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Background Task Processing
# ---------------------------------------------------------------------------

def _process_document_task(
    task_id: str,
    file_path: Path,
    filename: str,
    pages: str | None,
    ocr: bool,
    ocr_lang: str,
    source_doc: str,
):
    """Background thread that processes a document and reports progress."""
    try:
        _set_progress(task_id, status="processing", phase="parsing",
                      message="正在解析文档...")

        # Parse
        parser = _get_parser(file_path)
        if not parser:
            _set_progress(task_id, status="error",
                          error=f"Unsupported file type: {filename}")
            return

        # Pass progress callback for PDF parser
        if isinstance(parser, PdfParser):
            def pdf_progress(info: dict):
                _set_progress(task_id, **info)

            text = parser.parse(file_path, pages=pages, ocr_enabled=ocr,
                                ocr_lang=ocr_lang, progress_callback=pdf_progress)
        else:
            text = parser.parse(file_path)

        # Check if we have pages info for PDF
        if isinstance(parser, PdfParser) and pages:
            from doc2graph.parser.pdf_parser import parse_page_range
            try:
                total = parser.get_page_count(file_path)
                selected = parse_page_range(pages, total)
                _set_progress(task_id,
                              page_current=len(selected),
                              page_total=len(selected))
            except Exception:
                pass

        _set_progress(task_id, phase="extracting",
                      message="正在提取实体和关系...")

        # Extract
        doc = Document.from_path(file_path)
        actual_source_doc = source_doc.strip() if source_doc else doc.source_doc_id

        extractor = LlmExtractor(
            api_key=config.llm.api_key,
            model=config.llm.model,
            api_base=config.llm.api_base,
            max_entities=config.extraction.max_entities,
            max_relations=config.extraction.max_relations,
            temperature=config.extraction.temperature,
        )
        entities, relations = extractor.extract(text, source_doc=actual_source_doc)

        _set_progress(task_id, phase="importing",
                      message="正在导入到图谱...")

        # Import
        client = _get_client()
        importer = GraphImporter(client)
        counts = importer.import_graph(entities, relations)

        _set_progress(task_id, status="done", phase="done",
                      message="处理完成!",
                      result={
                          "entities": counts.get("entities", 0),
                          "relations": counts.get("relations", 0),
                          "source_doc": actual_source_doc,
                      })

    except Exception as e:
        logger.error(f"Task {task_id} error: {e}")
        _set_progress(task_id, status="error", error=str(e))
    finally:
        # Clean up uploaded file
        if file_path.exists():
            file_path.unlink()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/process")
async def process_document(
    file: UploadFile = File(...),
    pages: str | None = Form(None),
    ocr: str = Form("true"),
    ocr_lang: str = Form("chi_sim+eng"),
    source_doc: str | None = Form(None),
):
    """Upload and process a single document asynchronously.

    Returns a task_id that can be used to poll progress via /api/progress/{task_id}.

    Args:
        file: The document file (PDF, MD, TXT, DOCX).
        pages: Page range for PDF files (e.g. "1,3,5-10", "1", "all").
               1-based page numbers. Default is all pages.
        ocr: Enable OCR for PDF images. Default is True.
        ocr_lang: Tesseract OCR language (e.g. "chi_sim+eng"). Default is Chinese + English.
        source_doc: Custom source document identifier. Defaults to filename without extension.
    """
    if config is None or not config.llm.api_key:
        raise HTTPException(status_code=500, detail="LLM API key not configured")

    # Save uploaded file temporarily
    upload_dir = Path("/tmp/doc2graph_uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename

    content = await file.read()
    file_path.write_bytes(content)

    # Create task
    task_id = str(uuid.uuid4())[:8]
    actual_source_doc = source_doc.strip() if source_doc else file.filename.rsplit(".", 1)[0]
    ocr_enabled = ocr.lower() in ("true", "1", "yes", "on")

    _set_progress(task_id,
                  status="pending",
                  phase="upload",
                  message="任务已创建，等待处理...",
                  filename=file.filename,
                  pages=pages or "all",
                  ocr=ocr_enabled,
                  page_current=0,
                  page_total=None)

    # Start background thread
    thread = threading.Thread(
        target=_process_document_task,
        args=(task_id, file_path, file.filename, pages, ocr_enabled, ocr_lang, actual_source_doc),
        daemon=True,
    )
    thread.start()

    logger.info(f"Task {task_id} started for {file.filename} (pages={pages or 'all'})")

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "任务已提交，正在处理中",
        "filename": file.filename,
        "pages": pages or "all",
    }


@app.get("/api/progress/{task_id}")
async def get_progress(task_id: str):
    """Get the progress of a document processing task.

    Poll this endpoint to track progress. Returns progress info including:
    - status: "pending" | "processing" | "done" | "error"
    - phase: "parsing" | "extracting" | "importing" | "done"
    - page_current: current page being processed
    - page_total: total pages to process
    - message: human-readable status message
    - result: final result when status is "done"
    - error: error message when status is "error"
    """
    progress = _get_progress(task_id)
    if progress is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return progress


@app.get("/api/cleanup")
async def cleanup_tasks():
    """Clean up old completed tasks."""
    _cleanup_old_tasks()
    return {"message": "Cleanup complete"}


@app.get("/api/documents")
async def list_documents():
    """List all processed documents in the graph."""
    try:
        client = _get_client()
        results = client.run_query(
            "MATCH (n) WHERE n.source_doc IS NOT NULL "
            "RETURN n.source_doc AS doc_name, "
            "count(DISTINCT n) AS entity_count, "
            "collect(DISTINCT n.type) AS types, "
            "min(n.created_at) AS first_seen, "
            "max(n.created_at) AS last_seen "
            "ORDER BY last_seen DESC"
        )
        return [
            {
                "doc_name": r.get("doc_name", ""),
                "entity_count": r.get("entity_count", 0),
                "type_count": len(r.get("types", [])),
                "types": r.get("types", []),
                "first_seen": r.get("first_seen", ""),
                "last_seen": r.get("last_seen", ""),
            }
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/merge", response_model=MergeResponse)
async def merge_graph():
    """Merge entities across all documents in the graph."""
    try:
        client = _get_client()
        merger = GraphMerger(client)
        merger.merge_in_neo4j()
        stats = client.get_stats()
        return MergeResponse(
            merged=0,
            total_nodes=stats["nodes"],
            total_relationships=stats["relationships"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get graph statistics."""
    try:
        client = _get_client()
        stats = client.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/entities", response_model=list[EntityItem])
async def list_entities(limit: int = Query(50, le=10000)):
    """List entities in the graph."""
    try:
        client = _get_client()
        query = GraphQuery(client)
        results = query.list_entities(limit=limit)
        return [
            EntityItem(
                name=r.get("name", ""),
                type=r.get("type", ""),
                source=r.get("source"),
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/relations", response_model=list[RelationItem])
async def list_relations(limit: int = Query(50, le=10000)):
    """List relations in the graph."""
    try:
        client = _get_client()
        query = GraphQuery(client)
        results = query.list_relations(limit=limit)
        return [
            RelationItem(
                source=r.get("source", ""),
                type=r.get("type", ""),
                target=r.get("target", ""),
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    """Execute a Cypher query."""
    try:
        client = _get_client()
        results = client.run_query(req.cypher)
        return QueryResponse(results=results, count=len(results))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/export")
async def export_graph(fmt: str = Query("json", pattern="^(json|cypher)$")):
    """Export the graph as JSON or Cypher."""
    try:
        client = _get_client()
        query = GraphQuery(client)

        if fmt == "json":
            data = query.export_json()
            return data
        else:
            cypher = query.export_cypher()
            return {"cypher": cypher}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        client = _get_client()
        stats = client.get_stats()
        return {"status": "healthy", "graph": stats}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/api/pdf-info")
async def pdf_info(file: UploadFile = File(...)):
    """Get PDF file information (page count, etc.) without full processing."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    upload_dir = Path("/tmp/doc2graph_uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename

    content = await file.read()
    file_path.write_bytes(content)

    try:
        parser = PdfParser()
        page_count = parser.get_page_count(file_path)
        return {"filename": file.filename, "pages": page_count, "type": "pdf"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path.exists():
            file_path.unlink()
