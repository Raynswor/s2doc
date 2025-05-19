import pytest

from src.s2doc.semantics import (
    Literal,
    SemanticEntity,
    SemanticKnowledgeGraph,
    SemanticRelationship,
    SemanticType,
)


def test_add_literal_property_success():
    network = SemanticKnowledgeGraph()
    entity = SemanticEntity(uri="http://example.org/entity1", label="Entity1")
    network.add_entity(entity)
    literal = Literal(value="Test Value", datatype="xsd:string")

    network.add_literal_property(
        entity_uri="http://example.org/entity1",
        prop_uri="http://example.org/property1",
        literal=literal,
    )

    assert (
        "http://example.org/property1"
        in network.entities["http://example.org/entity1"].literals
    )
    assert (
        network.entities["http://example.org/entity1"].literals[
            "http://example.org/property1"
        ]
        == literal
    )


def test_add_literal_property_entity_not_found():
    network = SemanticKnowledgeGraph()
    literal = Literal(value="Test Value", datatype="xsd:string")

    with pytest.raises(
        KeyError, match="Entity 'http://example.org/entity1' not in network"
    ):
        network.add_literal_property(
            entity_uri="http://example.org/entity1",
            prop_uri="http://example.org/property1",
            literal=literal,
        )


def test_semantic_entity_initialization():
    entity = SemanticEntity(
        uri="http://example.org/entity1",
        label="Entity1",
        type=SemanticType(uri="http://example.org/type1", label="Type1"),
        flags={"flag1": True},
        literals={"http://example.org/property1": Literal(value="Test Value")},
    )
    assert entity.uri == "http://example.org/entity1"
    assert entity.label == "Entity1"
    assert entity.type.uri == "http://example.org/type1"
    assert entity.flags["flag1"] is True
    assert "http://example.org/property1" in entity.literals


def test_semantic_entity_to_obj():
    entity = SemanticEntity(
        uri="http://example.org/entity1",
        label="Entity1",
        type=SemanticType(uri="http://example.org/type1", label="Type1"),
        flags={"flag1": True},
        literals={"http://example.org/property1": Literal(value="Test Value")},
    )
    obj = entity.to_obj()
    assert obj["uri"] == "http://example.org/entity1"
    assert obj["label"] == "Entity1"
    assert obj["type"] == "http://example.org/type1"
    assert obj["flags"]["flag1"] is True
    assert obj["literals"]["http://example.org/property1"]["value"] == "Test Value"


def test_semantic_network_add_entity():
    network = SemanticKnowledgeGraph()
    entity = SemanticEntity(uri="http://example.org/entity1", label="Entity1")
    network.add_entity(entity)
    assert "http://example.org/entity1" in network.entities


def test_semantic_network_add_entity_duplicate():
    network = SemanticKnowledgeGraph()
    entity = SemanticEntity(uri="http://example.org/entity1", label="Entity1")
    network.add_entity(entity)
    with pytest.raises(
        KeyError, match="Entity 'http://example.org/entity1' already in network"
    ):
        network.add_entity(entity)


def test_semantic_network_add_type():
    network = SemanticKnowledgeGraph()
    typ = SemanticType(uri="http://example.org/type1", label="Type1")
    network.add_type(typ)
    assert "http://example.org/type1" in network.available_types


def test_semantic_network_add_type_duplicate():
    network = SemanticKnowledgeGraph()
    typ = SemanticType(uri="http://example.org/type1", label="Type1")
    network.add_type(typ)
    with pytest.raises(
        KeyError, match="Type 'http://example.org/type1' already in network"
    ):
        network.add_type(typ)


def test_semantic_network_add_relationship():
    network = SemanticKnowledgeGraph()
    entity1 = SemanticEntity(uri="http://example.org/entity1", label="Entity1")
    entity2 = SemanticEntity(uri="http://example.org/entity2", label="Entity2")
    network.add_entity(entity1)
    network.add_entity(entity2)
    relationship = SemanticRelationship(
        label="relatedTo",
        head="http://example.org/entity1",
        tail="http://example.org/entity2",
    )
    network.add_relationship(relationship=relationship)
    assert relationship in network.relationships


def test_semantic_network_add_literal_property():
    network = SemanticKnowledgeGraph()
    entity = SemanticEntity(uri="http://example.org/entity1", label="Entity1")
    network.add_entity(entity)
    literal = Literal(value="Test Value", datatype="xsd:string")
    network.add_literal_property(
        entity_uri="http://example.org/entity1",
        prop_uri="http://example.org/property1",
        literal=literal,
    )
    assert (
        "http://example.org/property1"
        in network.entities["http://example.org/entity1"].literals
    )
    assert (
        network.entities["http://example.org/entity1"].literals[
            "http://example.org/property1"
        ]
        == literal
    )


def test_semantic_network_add_literal_property_entity_not_found():
    network = SemanticKnowledgeGraph()
    literal = Literal(value="Test Value", datatype="xsd:string")
    with pytest.raises(
        KeyError, match="Entity 'http://example.org/entity1' not in network"
    ):
        network.add_literal_property(
            entity_uri="http://example.org/entity1",
            prop_uri="http://example.org/property1",
            literal=literal,
        )
