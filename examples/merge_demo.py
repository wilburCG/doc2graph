"""Graph merge demo — show how entity resolution works across documents."""

from doc2graph.graph.merger import GraphMerger
from doc2graph.models.entity import Entity
from doc2graph.models.relation import Relation


def main():
    print("=== Doc2Graph Merge Demo ===\n")

    # Simulate entities from two different documents about the same topic
    doc1_entities = [
        Entity(name="Python", type="Technology", source_doc="lang_guide.md", properties={"description": "A programming language"}),
        Entity(name="Django", type="Technology", source_doc="lang_guide.md", properties={"description": "Web framework"}),
        Entity(name="Guido", type="Person", source_doc="lang_guide.md", properties={"description": "Creator of Python"}),
    ]
    doc1_relations = [
        Relation(source="Guido", target="Python", type="CREATED", source_doc="lang_guide.md"),
        Relation(source="Python", target="Django", type="POWERS", source_doc="lang_guide.md"),
    ]

    doc2_entities = [
        Entity(name="Python", type="Technology", source_doc="web_dev.md", properties={"version": "3.11"}),
        Entity(name="FastAPI", type="Technology", source_doc="web_dev.md", properties={"description": "Modern web framework"}),
        Entity(name="Flask", type="Technology", source_doc="web_dev.md", properties={"description": "Micro web framework"}),
    ]
    doc2_relations = [
        Relation(source="Python", target="FastAPI", type="POWERS", source_doc="web_dev.md"),
        Relation(source="Python", target="Flask", type="POWERS", source_doc="web_dev.md"),
    ]

    all_entities = doc1_entities + doc2_entities
    all_relations = doc1_relations + doc2_relations

    print(f"Before merge: {len(all_entities)} entities, {len(all_relations)} relations")

    # Merge
    merger = GraphMerger.__new__(GraphMerger)
    merger.strategy = "name_type"
    merger.threshold = 0.85
    merged_entities, merged_relations = merger.merge_all(all_entities, all_relations)

    print(f"After merge:  {len(merged_entities)} entities, {len(merged_relations)} relations\n")

    print("Merged entities:")
    for e in merged_entities:
        print(f"  - {e.name} ({e.type}) [sources: {e.source_doc}]")

    print("\nMerged relations:")
    for r in merged_relations:
        print(f"  - {r.source} -[{r.type}]-> {r.target}")

    # Check Python entity — should have both sources
    python = [e for e in merged_entities if e.name == "Python"][0]
    print(f"\nPython entity sources: {python.source_doc}")
    print(f"Python properties: {python.properties}")


if __name__ == "__main__":
    main()
