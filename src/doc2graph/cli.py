"""CLI entry point for Doc2Graph."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from doc2graph import __version__
from doc2graph.config import Config, DEFAULT_CONFIG_YAML
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

app = typer.Typer(
    name="doc2graph",
    help="Convert documents to Neo4j knowledge graphs using LLMs",
    add_completion=False,
)
console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PARSERS = [PdfParser, MdParser, TxtParser, DocxParser]


def _get_parser(file_path: Path):
    for parser_cls in PARSERS:
        if parser_cls.supports(file_path):
            return parser_cls()
    return None


def _load_config(config_path: str | None) -> Config:
    if config_path:
        return Config.load(config_path)
    # Try default locations
    for p in [Path("config.yaml"), Path.home() / ".doc2graph" / "config.yaml"]:
        if p.exists():
            return Config.load(p)
    return Config()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def version() -> None:
    """Show version."""
    console.print(f"doc2graph v{__version__}")


@app.command()
def init(
    path: str = typer.Argument("config.yaml", help="Path to write config file"),
) -> None:
    """Generate a default configuration file."""
    p = Path(path)
    if p.exists():
        console.print(f"[yellow]Config file already exists: {p}[/yellow]")
        return
    with open(p, "w", encoding="utf-8") as f:
        f.write(DEFAULT_CONFIG_YAML)
    console.print(f"[green]✓[/green] Config written to {p}")
    console.print("Edit it and set your API keys, then run [bold]doc2graph process[/bold]")


@app.command()
def process(
    path: str = typer.Argument(..., help="Path to a document file or directory"),
    pages: str = typer.Option(None, "--pages", "-p", help="PDF page range (e.g. '1,3,5-10', '1', 'all')"),
    ocr: bool = typer.Option(True, "--ocr/--no-ocr", help="Enable OCR for PDF images"),
    ocr_lang: str = typer.Option("chi_sim+eng", "--ocr-lang", help="OCR language (e.g. chi_sim+eng)"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Process directories recursively"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Process one or more documents: parse → extract → import to Neo4j."""
    _setup_logging(verbose)
    cfg = _load_config(config)

    if not cfg.llm.api_key:
        console.print("[red]Error: LLM API key not set. Configure it in config.yaml or set OPENAI_API_KEY.[/red]")
        raise typer.Exit(1)

    if not cfg.neo4j.password:
        console.print("[red]Error: Neo4j password not set. Configure it in config.yaml.[/red]")
        raise typer.Exit(1)

    # Resolve paths
    p = Path(path)
    if p.is_dir():
        files = list(p.rglob("*") if recursive else p.iterdir())
        files = [f for f in files if f.is_file()]
    else:
        files = [p]

    # Filter to supported types
    supported = set()
    for parser_cls in PARSERS:
        supported.update(parser_cls.supported_extensions())
    files = [f for f in files if f.suffix.lower() in supported]

    if not files:
        console.print(f"[yellow]No supported documents found in {path}[/yellow]")
        return

    console.print(f"\n[bold]Processing {len(files)} document(s)...[/bold]\n")

    neo4j = Neo4jClient(cfg.neo4j.uri, cfg.neo4j.username, cfg.neo4j.password, cfg.neo4j.database)
    extractor = LlmExtractor(
        api_key=cfg.llm.api_key,
        model=cfg.llm.model,
        api_base=cfg.llm.api_base,
        max_entities=cfg.extraction.max_entities,
        max_relations=cfg.extraction.max_relations,
        temperature=cfg.extraction.temperature,
    )
    importer = GraphImporter(neo4j)

    total_entities = 0
    total_relations = 0

    try:
        for file_path in files:
            console.print(f"  📄 [cyan]{file_path.name}[/cyan]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                # Parse
                task = progress.add_task("Parsing...", total=None)
                parser = _get_parser(file_path)
                if not parser:
                    progress.update(task, description="[red]Unsupported format[/red]")
                    continue

                # Pass page and OCR options for PDF parser
                if isinstance(parser, PdfParser):
                    text = parser.parse(file_path, pages=pages, ocr_enabled=ocr, ocr_lang=ocr_lang)
                else:
                    text = parser.parse(file_path)
                progress.update(task, description="[green]✓ Parsed[/green]")

                # Extract
                task = progress.add_task("Extracting entities & relations...", total=None)
                doc = Document.from_path(file_path)
                entities, relations = extractor.extract(text, source_doc=doc.source_doc_id)
                progress.update(task, description=f"[green]✓ {len(entities)} entities, {len(relations)} relations[/green]")

                # Import
                task = progress.add_task("Importing to Neo4j...", total=None)
                counts = importer.import_graph(entities, relations)
                progress.update(task, description="[green]✓ Imported[/green]")

            total_entities += counts.get("entities", 0)
            total_relations += counts.get("relations", 0)
            console.print()

        stats = neo4j.get_stats()
        console.print(f"\n[bold green]Done![/bold green] {total_entities} entities, {total_relations} relations processed")
        console.print(f"Graph now has: [bold]{stats['nodes']}[/bold] nodes, [bold]{stats['relationships']}[/bold] relationships")

    finally:
        neo4j.close()


@app.command()
def merge(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    all_docs: bool = typer.Option(False, "--all", help="Merge all documents in the graph"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Merge entities and relations across documents (entity resolution)."""
    _setup_logging(verbose)
    cfg = _load_config(config)

    if not cfg.neo4j.password:
        console.print("[red]Error: Neo4j password not set.[/red]")
        raise typer.Exit(1)

    neo4j = Neo4jClient(cfg.neo4j.uri, cfg.neo4j.username, cfg.neo4j.password, cfg.neo4j.database)
    merger = GraphMerger(neo4j, strategy=cfg.merge.strategy, threshold=cfg.merge.threshold)

    console.print("\n[bold]Merging graph entities...[/bold]")
    result = merger.merge_in_neo4j()
    console.print(f"[green]✓[/green] Merge complete: {result}")

    stats = neo4j.get_stats()
    console.print(f"Graph stats: {stats['nodes']} nodes, {stats['relationships']} relationships")
    neo4j.close()


@app.command()
def query(
    cypher: str = typer.Argument(..., help="Cypher query to execute"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Execute a Cypher query against the graph."""
    _setup_logging(verbose)
    cfg = _load_config(config)

    neo4j = Neo4jClient(cfg.neo4j.uri, cfg.neo4j.username, cfg.neo4j.password, cfg.neo4j.database)

    try:
        results = neo4j.run_query(cypher)
        if not results:
            console.print("[yellow]No results.[/yellow]")
            return

        table = Table()
        for key in results[0].keys():
            table.add_column(key, style="cyan")
        for row in results:
            table.add_row(*[str(v)[:50] for v in row.values()])
        console.print(table)
    finally:
        neo4j.close()


@app.command()
def export(
    output: str = typer.Option("graph.json", "--output", "-o", help="Output file path"),
    fmt: str = typer.Option("json", "--format", "-f", help="Output format: json or cypher"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Export the graph to JSON or Cypher format."""
    _setup_logging(verbose)
    cfg = _load_config(config)

    neo4j = Neo4jClient(cfg.neo4j.uri, cfg.neo4j.username, cfg.neo4j.password, cfg.neo4j.database)
    graph_query = GraphQuery(neo4j)

    try:
        if fmt == "json":
            data = graph_query.export_json()
            with open(output, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        elif fmt == "cypher":
            cypher = graph_query.export_cypher()
            with open(output, "w", encoding="utf-8") as f:
                f.write(cypher)
        else:
            console.print(f"[red]Unknown format: {fmt}. Use 'json' or 'cypher'.[/red]")
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Exported to {output}")
    finally:
        neo4j.close()


@app.command()
def stats(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Show graph statistics."""
    cfg = _load_config(config)

    if not cfg.neo4j.password:
        console.print("[red]Error: Neo4j password not set.[/red]")
        raise typer.Exit(1)

    neo4j = Neo4jClient(cfg.neo4j.uri, cfg.neo4j.username, cfg.neo4j.password, cfg.neo4j.database)
    try:
        s = neo4j.get_stats()
        console.print(f"\n[bold]Graph Statistics:[/bold]")
        console.print(f"  Nodes:          [cyan]{s['nodes']}[/cyan]")
        console.print(f"  Relationships:  [cyan]{s['relationships']}[/cyan]")
    finally:
        neo4j.close()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on changes"),
) -> None:
    """Start the API server with web frontend."""
    import uvicorn

    console.print(f"\n[bold]Starting Doc2Graph API server...[/bold]")
    console.print(f"  URL:  http://{host}:{port}")
    console.print(f"  API:  http://{host}:{port}/docs")
    console.print()

    uvicorn.run(
        "doc2graph.api:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
