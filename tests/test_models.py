"""Tests for entity and relation models."""

from doc2graph.models.entity import Entity
from doc2graph.models.relation import Relation


class TestEntity:
    def test_create_entity(self):
        e = Entity(name="Python", type="Technology", source_doc="doc1")
        assert e.name == "Python"
        assert e.type == "Technology"
        assert e.source_doc == "doc1"

    def test_neo4j_label(self):
        e = Entity(name="X", type="Person")
        assert e.neo4j_label() == "Entity_Person"

    def test_neo4j_props(self):
        e = Entity(name="X", type="Person", properties={"description": "A person"}, source_doc="d1")
        props = e.model_dump_for_neo4j()
        assert props["name"] == "X"
        assert props["description"] == "A person"
        assert props["source_doc"] == "d1"

    def test_equality(self):
        a = Entity(name="Python", type="Technology")
        b = Entity(name="Python", type="Technology")
        c = Entity(name="Python", type="Language")
        assert a == b
        assert a != c

    def test_hash(self):
        a = Entity(name="Python", type="Technology")
        b = Entity(name="Python", type="Technology")
        assert hash(a) == hash(b)


class TestRelation:
    def test_create_relation(self):
        r = Relation(source="Alice", target="Python", type="CREATED", source_doc="doc1")
        assert r.source == "Alice"
        assert r.target == "Python"
        assert r.type == "CREATED"

    def test_neo4j_props(self):
        r = Relation(source="A", target="B", type="RELATED_TO", source_doc="d1")
        props = r.model_dump_for_neo4j()
        assert props["source_doc"] == "d1"
