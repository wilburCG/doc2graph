"""Quick start example — process a document and import to Neo4j.

Usage:
    export OPENAI_API_KEY="your-key"
    python examples/quick_start.py docs/sample.md
"""

import sys
from pathlib import Path

from doc2graph.config import Config
from doc2graph.extractor.llm_extractor import LlmExtractor
from doc2graph.graph.importer import GraphImporter
from doc2graph.graph.neo4j_client import Neo4jClient
from doc2graph.parser.md_parser import MdParser
from doc2graph.parser.pdf_parser import PdfParser


def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_start.py <document_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    cfg = Config.load()

    # Parse
    print(f"Parsing {file_path.name}...")
    parser = MdParser() if file_path.suffix == ".md" else PdfParser()
    text = parser.parse(file_path)
    print(f"  Extracted {len(text)} characters")

    # Extract
    print("Extracting entities and relations...")
    extractor = LlmExtractor(
        api_key=cfg.llm.api_key,
        model=cfg.llm.model,
        api_base=cfg.llm.api_base,
    )
    entities, relations = extractor.extract(text, source_doc=file_path.stem)
    print(f"  Found {len(entities)} entities, {len(relations)} relations")

    # Import
    print("Importing to Neo4j...")
    neo4j = Neo4jClient(cfg.neo4j.uri, cfg.neo4j.username, cfg.neo4j.password)
    importer = GraphImporter(neo4j)
    counts = importer.import_graph(entities, relations)
    print(f"  Imported: {counts}")

    stats = neo4j.get_stats()
    print(f"\nGraph: {stats['nodes']} nodes, {stats['relationships']} relationships")
    neo4j.close()


if __name__ == "__main__":
    main()
