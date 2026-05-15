"""FastAPI server for Doc2Graph.

Provides REST API endpoints for document processing, graph queries, and export.

Usage:
    uvicorn doc2graph.api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
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
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/process", response_model=ProcessResponse)
async def process_document(file: UploadFile = File(...)):
    """Upload and process a single document."""
    if config is None or not config.llm.api_key:
        raise HTTPException(status_code=500, detail="LLM API key not configured")

    # Save uploaded file temporarily
    upload_dir = Path("/tmp/doc2graph_uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename

    content = await file.read()
    file_path.write_bytes(content)

    try:
        # Parse
        parser = _get_parser(file_path)
        if not parser:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
        text = parser.parse(file_path)

        # Extract
        doc = Document.from_path(file_path)
        extractor = LlmExtractor(
            api_key=config.llm.api_key,
            model=config.llm.model,
            api_base=config.llm.api_base,
            max_entities=config.extraction.max_entities,
            max_relations=config.extraction.max_relations,
            temperature=config.extraction.temperature,
        )
        entities, relations = extractor.extract(text, source_doc=doc.source_doc_id)

        # Import
        client = _get_client()
        importer = GraphImporter(client)
        counts = importer.import_graph(entities, relations)

        return ProcessResponse(
            entities=counts.get("entities", 0),
            relations=counts.get("relations", 0),
            source_doc=doc.source_doc_id,
        )
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up
        if file_path.exists():
            file_path.unlink()


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
async def list_entities(limit: int = Query(50, le=500)):
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
async def list_relations(limit: int = Query(50, le=500)):
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
