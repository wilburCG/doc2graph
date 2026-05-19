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


# ---------------------------------------------------------------------------
# Graph RAG Integration Endpoints (for MaxKB)
# ---------------------------------------------------------------------------

class GraphSearchRequest(BaseModel):
    """Request to search the knowledge graph."""
    query: str = Field(..., description="Search query in natural language")
    limit: int = Field(10, description="Maximum number of results")
    min_score: float = Field(0.5, description="Minimum similarity score (0-1)")


class GraphSearchResponse(BaseModel):
    """Response from graph search."""
    entities: list[dict] = Field(default_factory=list, description="Matched entities")
    relations: list[dict] = Field(default_factory=list, description="Relevant relations")
    context_text: str = Field("", description="Context text for RAG")
    summary: str = Field("", description="Brief summary of the graph context")


class GraphQaRequest(BaseModel):
    """Request for graph-based question answering."""
    question: str = Field(..., description="Question to answer")
    use_llm: bool = Field(True, description="Use LLM to generate answer from context")


class GraphQaResponse(BaseModel):
    """Response from graph QA."""
    question: str = Field("", description="Original question")
    answer: str = Field("", description="Generated answer")
    context: dict = Field(default_factory=dict, description="Graph context used")
    sources: list[str] = Field(default_factory=list, description="Source documents")
    confidence: float = Field(0.0, description="Confidence score (0-1)")


class EntitySearchRequest(BaseModel):
    """Request to search entities."""
    name: str = Field("", description="Entity name to search (supports fuzzy matching)")
    type: str = Field("", description="Entity type filter")
    source_doc: str = Field("", description="Source document filter")
    limit: int = Field(50, description="Maximum results")


def _extract_keywords_chinese(q: str) -> str:
    """Extract keywords from Chinese question."""
    keywords = q
    
    # Remove question markers
    keywords = keywords.rstrip("？?！!。,.， ")
    keywords = keywords.rstrip("是什么").rstrip("是啥").rstrip("是谁")
    
    # Remove common question prefixes (longest first)
    prefixes = [
        "什么是", "什么叫", "啥是", "谁是",
        "解释一下", "介绍一下", "说明一下",
        "列出所有", "列出", "所有",
        "请问", "我想知道", "想了解",
        "哪些是", "哪些", "什么",
    ]
    
    for p in sorted(prefixes, key=len, reverse=True):
        if keywords.startswith(p):
            keywords = keywords[len(p):]
            break
    
    # Remove suffixes for relation questions
    suffixes = ["的关系", "的联系", "的关联", "是什么", "是啥"]
    for s in suffixes:
        if keywords.endswith(s):
            keywords = keywords[:-len(s)]
    
    # Remove relation middle words for entity extraction
    relation_words = ["和", "与", "跟", "之间", "相关的", "有关的", "的"]
    for w in relation_words:
        keywords = keywords.replace(w, " ")
    
    # Clean up whitespace - return first meaningful keyword
    parts = [p.strip() for p in keywords.split() if p.strip() and len(p.strip()) >= 2]
    
    if parts:
        return parts[0]  # Return the first meaningful keyword
    
    # Fallback: use first 2-4 chars
    return q[:4] if len(q) >= 4 else q.strip()


def _generate_cypher_from_query(nl_query: str) -> str:
    """Generate Cypher query from natural language (Chinese supported)."""
    q = nl_query.lower().strip()
    
    # Extract main entity/keywords
    entity_name = _extract_keywords_chinese(q)
    
    # Pattern: "X 和 Y 的关系" / "X与Y的关系"
    if any(k in q for k in ["和", "与", "跟"]) and any(k in q for k in ["关系", "联系", "关联"]):
        # Try to extract two entities
        import re
        # Split by common connectors
        separators = ["和", "与", "跟", "之间的"]
        parts = None
        for sep in separators:
            if sep in entity_name:
                parts = entity_name.split(sep, 1)
                break
        if parts and len(parts) == 2:
            a, b = parts[0].strip(), parts[1].strip()
            if a and b:
                return f"""
                MATCH (a)-[r]-(b)
                WHERE (toLower(a.name) CONTAINS '{a}' AND toLower(b.name) CONTAINS '{b}')
                   OR (toLower(a.name) CONTAINS '{b}' AND toLower(b.name) CONTAINS '{a}')
                RETURN a, type(r) AS relation, b
                LIMIT 15
                """
    
    # Pattern: "X 相关的" / "与 X 关联的"
    if any(k in q for k in ["相关", "关联", "有关"]):
        return f"""
        MATCH (n)-[r]->(m)
        WHERE toLower(n.name) CONTAINS '{entity_name}' OR toLower(m.name) CONTAINS '{entity_name}'
        RETURN n, type(r) AS relation, m
        LIMIT 20
        """
    
    # Pattern: "列出所有 X" / "所有 X 实体" / "什么是 X" / "谁是 X"
    if (any(k in q for k in ["列出", "所有", "什么是", "啥是", "谁是", "哪些"]) or
        q.endswith(("是什么", "是啥", "是谁"))):
        
        # If contains "类型" or "种类" or "类", search by type
        if any(k in q for k in ["类型", "种类", "类别的", "分类"]):
            return f"""
            MATCH (n)
            WHERE toLower(n.type) CONTAINS '{entity_name}' OR toLower(n.name) CONTAINS '{entity_name}'
            RETURN n.name AS name, n.type AS type, n.source_doc AS source
            ORDER BY name
            LIMIT 50
            """
        
        # Default entity search with relations
        return f"""
        MATCH (n)
        WHERE toLower(n.name) CONTAINS '{entity_name}'
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n AS entity, type(r) AS rel_type, m AS target
        LIMIT 20
        """
    
    # Default: fuzzy search for entities with relations
    search_term = entity_name if len(entity_name) >= 2 else q[:4] if len(q) >= 4 else q
    return f"""
    MATCH (n)
    WHERE toLower(n.name) CONTAINS '{search_term}'
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN n AS entity, type(r) AS rel_type, m AS target
    LIMIT 15
    """


def _format_graph_context(entities: list[dict], relations: list[dict]) -> str:
    """Format graph results into context text for RAG."""
    lines = []
    lines.append("【知识图谱检索结果】")
    lines.append("")
    
    if entities:
        lines.append("实体信息：")
        for e in entities[:10]:
            name = e.get("name", "")
            etype = e.get("type", "")
            source = e.get("source", "") or e.get("source_doc", "")
            if name:
                source_info = f" (来源: {source})" if source else ""
                lines.append(f"  - {name} [{etype}]{source_info}")
        lines.append("")
    
    if relations:
        lines.append("关系信息：")
        for r in relations[:15]:
            src = r.get("source", "") or r.get("from", "")
            rel = r.get("type", "") or r.get("relation", "")
            tgt = r.get("target", "") or r.get("to", "")
            if src and rel and tgt:
                lines.append(f"  - {src} → [{rel}] → {tgt}")
        lines.append("")
    
    lines.append("请基于以上知识图谱信息回答用户问题。")
    return "\n".join(lines)


@app.post("/api/graph/search", response_model=GraphSearchResponse)
async def graph_search(req: GraphSearchRequest):
    """Search the knowledge graph using natural language.
    
    This endpoint performs semantic search on the graph and returns
    relevant entities and relations, formatted as RAG context.
    
    Args:
        query: Natural language search query
        limit: Maximum number of results
        min_score: Minimum similarity threshold
    
    Returns:
        GraphSearchResponse with entities, relations, and context text
    """
    try:
        client = _get_client()
        query = req.query.strip()
        
        # Generate and execute Cypher
        cypher = _generate_cypher_from_query(query)
        results = client.run_query(cypher)
        
        # Extract unique entities and relations
        entities_dict = {}
        relations_dict = {}
        
        for row in results:
            # Handle entity search results
            if "name" in row and "type" in row:
                key = f"{row['name']}:{row['type']}"
                if key not in entities_dict:
                    entities_dict[key] = {
                        "name": row.get("name", ""),
                        "type": row.get("type", ""),
                        "source": row.get("source", ""),
                    }
                continue
            
            # Handle entity-relation-entity results
            entity = row.get("entity", {}) or row.get("a", {})
            target = row.get("target", {}) or row.get("b", {})
            rel_type = row.get("rel_type", "") or row.get("relation", "")
            
            if isinstance(entity, dict) and entity.get("name"):
                key = f"{entity['name']}:{entity.get('type', '')}"
                if key not in entities_dict:
                    entities_dict[key] = {
                        "name": entity.get("name", ""),
                        "type": entity.get("type", ""),
                        "source": entity.get("source_doc", ""),
                    }
            
            if isinstance(target, dict) and target.get("name"):
                key = f"{target['name']}:{target.get('type', '')}"
                if key not in entities_dict:
                    entities_dict[key] = {
                        "name": target.get("name", ""),
                        "type": target.get("type", ""),
                        "source": target.get("source_doc", ""),
                    }
            
            if rel_type and isinstance(entity, dict) and isinstance(target, dict):
                src_name = entity.get("name", "")
                tgt_name = target.get("name", "")
                if src_name and tgt_name:
                    rel_key = f"{src_name}:{rel_type}:{tgt_name}"
                    if rel_key not in relations_dict:
                        relations_dict[rel_key] = {
                            "source": src_name,
                            "type": rel_type,
                            "target": tgt_name,
                        }
        
        entities = list(entities_dict.values())[:req.limit]
        relations = list(relations_dict.values())[:req.limit]
        
        # Build summary
        summary = f"找到 {len(entities)} 个相关实体和 {len(relations)} 个相关关系"
        if entities:
            entity_names = [e["name"] for e in entities[:5]]
            summary += f"，包括：{', '.join(entity_names)}"
            if len(entities) > 5:
                summary += f" 等"
        
        return GraphSearchResponse(
            entities=entities,
            relations=relations,
            context_text=_format_graph_context(entities, relations),
            summary=summary,
        )
    except Exception as e:
        logger.error(f"Graph search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/graph/qa", response_model=GraphQaResponse)
async def graph_qa(req: GraphQaRequest):
    """Question answering using the knowledge graph.
    
    This endpoint searches the graph for relevant information and optionally
    uses LLM to generate a natural language answer.
    
    Args:
        question: The question to answer
        use_llm: Whether to use LLM to generate the answer
    
    Returns:
        GraphQaResponse with answer, context, and sources
    """
    try:
        # First, search the graph
        search_req = GraphSearchRequest(query=req.question, limit=15)
        search_result = await graph_search(search_req)
        
        # Collect sources
        sources_set = set()
        for e in search_result.entities:
            if e.get("source"):
                sources_set.add(e["source"])
        sources = list(sources_set)
        
        # Calculate confidence based on number of matches
        confidence = min(1.0, (len(search_result.entities) * 0.1 + len(search_result.relations) * 0.05))
        
        # Generate answer
        answer = ""
        if req.use_llm and search_result.context_text:
            try:
                if config is None or not config.llm.api_key:
                    answer = search_result.context_text
                else:
                    # Use LLM to generate answer from context
                    from openai import OpenAI
                    client = OpenAI(
                        api_key=config.llm.api_key,
                        base_url=config.llm.api_base,
                    )
                    prompt = f"""你是一个知识图谱问答助手。请基于以下知识图谱检索结果，用简洁的中文回答用户问题。

{search_result.context_text}

用户问题：{req.question}

要求：
1. 只基于提供的图谱信息回答，不要编造信息
2. 如果信息不足，明确说明无法找到相关信息
3. 回答要简洁明了
"""
                    response = client.chat.completions.create(
                        model=config.llm.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                    )
                    answer = response.choices[0].message.content or ""
            except Exception as llm_err:
                logger.warning(f"LLM answer generation failed: {llm_err}")
                answer = search_result.context_text
        else:
            answer = search_result.context_text
        
        return GraphQaResponse(
            question=req.question,
            answer=answer,
            context={
                "entities": search_result.entities,
                "relations": search_result.relations,
                "summary": search_result.summary,
            },
            sources=sources,
            confidence=confidence,
        )
    except Exception as e:
        logger.error(f"Graph QA error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/graph/entities")
async def search_entities(req: EntitySearchRequest):
    """Search entities with filters.
    
    Args:
        name: Entity name search (fuzzy match)
        type: Entity type filter (exact match)
        source_doc: Source document filter
        limit: Maximum results
    """
    try:
        client = _get_client()
        
        conditions = []
        params = {}
        
        if req.name:
            conditions.append("toLower(n.name) CONTAINS $name")
            params["name"] = req.name.lower()
        if req.type:
            conditions.append("n.type = $type")
            params["type"] = req.type
        if req.source_doc:
            conditions.append("n.source_doc = $source_doc")
            params["source_doc"] = req.source_doc
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        cypher = f"""
        MATCH (n)
        {where_clause}
        RETURN n.name AS name, n.type AS type, n.source_doc AS source_doc
        ORDER BY name
        LIMIT $limit
        """
        params["limit"] = req.limit
        
        results = client.run_query(cypher, params)
        return results
    except Exception as e:
        logger.error(f"Entity search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
