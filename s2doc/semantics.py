from pathlib import Path

from owlready2 import Ontology, get_ontology

from .errors import LoadFromDictError


def extract_label_from_uri(uri: str) -> str:
    if "#" in uri:
        return uri.split("#")[-1]
    elif "/" in uri:
        return uri.split("/")[-1]
    return uri


class SemanticType:
    def __init__(self, uri: str, label: str | None = None) -> None:
        self.uri: str = uri
        self.label: str = label or extract_label_from_uri(self.uri)

    def __str__(self):
        return f"<{self.uri}>"

    def to_obj(self):
        return {
            "label": self.label,
            "uri": self.uri,
        }

    @classmethod
    def from_dict(cls, d: dict):
        try:
            return cls(d["uri"], d.get("label"))
        except KeyError as e:
            raise LoadFromDictError(cls.__name__, str(e))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SemanticType):
            return self.uri == other.uri
        return False


class SemanticEntity:
    # equivalent to instances in ontology
    def __init__(
        self,
        uri: str,
        label: str,
        type: str | SemanticType | None = None,
        flags: dict | None = None,
    ) -> None:
        self.uri: str = uri
        self.label: str = label
        self.type: str | SemanticType | None = type
        self.flags: dict[str, bool | str | int] = flags or {}

    def __post_init__(self):
        if not self.label:
            self.label = extract_label_from_uri(self.uri)

    @classmethod
    def from_dict(cls, d: dict):
        try:
            return cls(
                d["uri"], d.get("label") or "", d.get("type"), d.get("flags") or {}
            )
        except KeyError as e:
            raise LoadFromDictError(cls.__name__, str(e))

    def __str__(self):
        return f"{self.label}<{self.type}>"

    def to_obj(self):
        ret: dict[str, str | dict | None] = {
            "uri": self.uri,
            "label": self.label,
            "type": self.type.uri if isinstance(self.type, SemanticType) else self.type,
        }
        if self.flags:
            ret["flags"] = self.flags
        return ret

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticEntity):
            return False
        return (
            self.uri == other.uri
            and self.type == other.type
            and self.flags == other.flags
        )


class SemanticRelationship:
    # similar to relationships in ontology
    # but can only exist between two semantic entities not types
    def __init__(
        self, label: str, head: str | SemanticEntity, tail: str | SemanticEntity
    ):
        self.label: str = label
        self.head: str | SemanticEntity = head
        self.tail: str | SemanticEntity = tail

    def __str__(self):
        return f"{self.label}: {self.head} -> {self.tail}"

    def to_obj(self):
        return {
            "label": self.label,
            "head": self.head.label
            if isinstance(self.head, SemanticEntity)
            else self.head,
            "tail": self.tail.label
            if isinstance(self.tail, SemanticEntity)
            else self.tail,
        }

    @classmethod
    def from_dict(cls, d: dict):
        try:
            return cls(d.get("label") or "", d.get("head") or "", d.get("tail") or "")
        except KeyError as e:
            raise LoadFromDictError(cls.__name__, str(e))


class SemanticNetwork:
    """
    Contains all semantic entities and relationships between them in the document.
    Types are not defined here.

    Types -> Classes
    Entities -> Instances
    """

    def __init__(
        self,
        entities: dict[str, SemanticEntity] | None = None,
        relationships: list[SemanticRelationship] | None = None,
        available_types: dict[str, SemanticType] | None = None,
        ontology_path: str | Path | None = None,
    ):
        self.entities: dict[str, SemanticEntity] = entities or {}
        self.relationships: list[SemanticRelationship] = relationships or []
        self.available_types: dict[str, SemanticType] = available_types or {}
        if ontology_path:
            self.src_ontology: Ontology = self.load_ontology(ontology_path)
            self.resolve_ontology()

    def load_ontology(self, path: str | Path | None = None) -> Ontology:
        if path is None:
            raise ValueError("Ontology path is required")
        if isinstance(path, Path):
            path = str(path)
        return get_ontology(path).load()

    def resolve_ontology(self):
        # load available types
        for cls in self.src_ontology.classes():
            typ = SemanticType(cls.iri)
            for inst in cls.instances():
                self.entities[inst.iri] = SemanticEntity(inst.iri, inst.name, typ)
            self.available_types[cls.iri] = typ

        self.resolve_relationships()

    def draw_graph(self):
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()
        for ent in self.entities.values():
            G.add_node(ent.label)
        for rel in self.relationships:
            if isinstance(rel.head, str):
                h = rel.head
            else:
                h = rel.head.label
            if isinstance(rel.tail, str):
                t = rel.tail
            else:
                t = rel.tail.label
            G.add_edge(h, t, label=rel.label)
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos, node_color="lightblue", edge_color="gray")
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()

    def add_relationship(self, relationship: SemanticRelationship):
        assert isinstance(relationship.head, (str, SemanticEntity))
        assert isinstance(relationship.tail, (str, SemanticEntity))
        self.relationships.append(relationship)

    def resolve_relationships(self):
        for rel in self.relationships:
            if isinstance(rel.head, str):
                rel.head = self.entities[rel.head]
            if isinstance(rel.tail, str):
                rel.tail = self.entities[rel.tail]

    def to_obj(self):
        return {
            "entities": [e.to_obj() for e in self.entities.values()],
            "relationships": [r.to_obj() for r in self.relationships],
            "types": [t.to_obj() for t in self.available_types.values()],
        }

    @classmethod
    def from_dict(cls, d: dict):
        try:
            return cls(
                entities={
                    e["label"]: SemanticEntity.from_dict(e) for e in d["entities"]
                },
                relationships=[
                    SemanticRelationship.from_dict(x) for x in d["relationships"]
                ],
                available_types={
                    t["uri"]: SemanticType.from_dict(t) for t in d["types"]
                },
            )
        except KeyError as e:
            raise LoadFromDictError(cls.__name__, str(e))
