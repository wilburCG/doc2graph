"""Tests for graph merger."""

from doc2graph.graph.merger import GraphMerger
from doc2graph.models.entity import Entity
from doc2graph.models.relation import Relation


class TestGraphMerger:
    def test_name_type_merge(self):
        entities = [
            Entity(name="Python", type="Technology", source_doc="doc1", properties={"description": "A language"}),
            Entity(name="Python", type="Technology", source_doc="doc2", properties={"version": "3.11"}),
            Entity(name="Java", type="Technology", source_doc="doc1"),
        ]
        merger = GraphMerger.__new__(GraphMerger)
        merger.strategy = "name_type"
        merger.threshold = 0.85
        merged = merger._name_type_merge(entities)
        assert len(merged) == 2

        python = [e for e in merged if e.name == "Python"][0]
        assert "doc1" in python.source_doc
        assert "doc2" in python.source_doc

    def test_fuzzy_similarity(self):
        assert GraphMerger._similarity("Python", "Python") == 1.0
        assert GraphMerger._similarity("Python", "python") == 1.0
        assert GraphMerger._similarity("", "x") == 0.0

    def test_relation_dedup(self):
        relations = [
            Relation(source="A", target="B", type="CREATED", source_doc="doc1"),
            Relation(source="A", target="B", type="CREATED", source_doc="doc2"),
            Relation(source="A", target="C", type="USED", source_doc="doc1"),
        ]
        merger = GraphMerger.__new__(GraphMerger)
        merged = merger.merge_relations(relations)
        assert len(merged) == 2
