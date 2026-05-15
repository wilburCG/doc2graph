# Doc2Graph

Convert documents to Neo4j knowledge graphs using LLMs.

## Features

- 📄 **Multi-format parsing** — PDF, Markdown, TXT, DOCX
- 🤖 **LLM-powered extraction** — Automatically identify entities and relations
- 🗂️ **Per-document graphs** — Each document becomes a subgraph with source tracking
- 🔗 **Graph merging** — Entity resolution across multiple documents
- 💻 **CLI tool** — Simple command-line interface
- 🔌 **Neo4j integration** — Direct import into Neo4j graph database

## Quick Start

### 1. Install

```bash
cd doc2graph
pip install -e ".[dev]"
```

### 2. Configure

```bash
doc2graph init  # generates config.yaml
```

Edit `config.yaml` with your Neo4j credentials and LLM API key.

### 3. Process Documents

```bash
# Single document
doc2graph process docs/report.pdf

# Directory of documents
doc2graph process docs/ --recursive

# Check graph stats
doc2graph stats
```

### 4. Query

```bash
# List entities
doc2graph query "MATCH (n) RETURN n.name, n.type LIMIT 20"

# Find connections
doc2graph query "MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name LIMIT 20"
```

### 5. Export

```bash
doc2graph export --format json --output graph.json
doc2graph export --format cypher --output graph.cypher
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `doc2graph init` | Generate default config |
| `doc2graph process <path>` | Process document(s) |
| `doc2graph merge` | Merge entities across documents |
| `doc2graph query <cypher>` | Run Cypher query |
| `doc2graph export` | Export graph (json/cypher) |
| `doc2graph stats` | Show graph statistics |

## Architecture

```
Document → Parser → LLM Extractor → Neo4j Importer → Graph
                                                    → Merger (multi-doc)
```

## Configuration

```yaml
neo4j:
  uri: bolt://localhost:7687
  username: neo4j
  password: your_password

llm:
  provider: openai
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}
  api_base: https://api.openai.com/v1

extraction:
  max_entities: 50
  max_relations: 100
  temperature: 0.1

merge:
  strategy: name_type
  threshold: 0.85
```

## Development

```bash
# Run tests
pytest

# Lint
ruff check src/

# Format
ruff format src/
```

## License

MIT — See [LICENSE](LICENSE) for details.
