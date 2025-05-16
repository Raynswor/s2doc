from pathlib import Path

from owlready2 import Ontology, get_ontology

from .errors import LoadFromDictError


def extract_label_from_uri(uri: str) -> str:
    if "#" in uri:
        return uri.split("#")[-1]
    elif "/" in uri:
        return uri.split("/")[-1]
    return uri


class Literal:
    _TYPE_MAP = {
        str: "xsd:string",
        int: "xsd:integer",
        float: "xsd:decimal",
        bool: "xsd:boolean",
    }

    def __init__(
        self, value: str | int | float | bool, datatype: str | None = None
    ) -> None:
        self.value = value
        # infer datatype if not explicitly provided
        self.datatype = (
            datatype
            if datatype is not None
            else self._TYPE_MAP.get(type(value), "xsd:string")
        )

    def __str__(self) -> str:
        # wrap strings in quotes, leave numbers/booleans unquoted
        if isinstance(self.value, str):
            lex = f'"{self.value}"'
        else:
            lex = str(self.value)
        return f"{lex}^^{self.datatype}"

    def to_obj(self) -> dict[str, str | int | float | bool]:
        return {
            "value": self.value,
            "datatype": self.datatype,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Literal":
        """
        Deserialize from dict, expecting keys "value" and optional "datatype".
        """
        try:
            return cls(d["value"], d.get("datatype"))
        except KeyError as e:
            raise LoadFromDictError(cls.__name__, str(e))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Literal):
            return False
        return self.value == other.value and self.datatype == other.datatype


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

    def __hash__(self) -> int:
        return hash(self.uri)


class SemanticEntity:
    # equivalent to instances in ontology
    def __init__(
        self,
        uri: str,
        label: str | None = None,
        type: str | SemanticType | None = None,
        flags: dict | None = None,
        literals: dict[str, Literal] | None = None,
    ) -> None:
        self.uri: str = uri
        self.label: str = label or extract_label_from_uri(uri)
        self.type: str | SemanticType | None = type
        self.flags: dict[str, bool | str | int] = flags or {}
        self.literals: dict[str, Literal] = literals or {}

    @classmethod
    def from_dict(cls, d: dict):
        try:
            literals = {
                k: Literal.from_dict(v) for k, v in d.get("literals", {}).items()
            }
            return cls(
                d["uri"],
                d.get("label") or "",
                d.get("type"),
                d.get("flags") or {},
                literals,
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
        if self.literals:
            ret["literals"] = {k: v.to_obj() for k, v in self.literals.items()}
        return ret

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticEntity):
            return False
        return (
            self.uri == other.uri
            and self.type == other.type
            and self.flags == other.flags
        )

    def __hash__(self) -> int:
        return hash(self.uri)


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

    def __hash__(self) -> int:
        return hash((self.label, self.head, self.tail))


class SemanticKnowledgeGraph:
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

        self.resolve_literals()
        self.resolve_relationships()

    def add_entity(self, entity: SemanticEntity):
        assert isinstance(entity, SemanticEntity)
        if entity.uri in self.entities:
            raise KeyError(f"Entity '{entity.uri}' already in network")
        self.entities[entity.uri] = entity
        if entity.type and isinstance(entity.type, SemanticType):
            self.available_types[entity.type.uri] = entity.type

    def add_type(self, t: SemanticType):
        assert isinstance(t, SemanticType)
        if t.uri in self.available_types:
            raise KeyError(f"Type '{t.uri}' already in network")
        self.available_types[t.uri] = t

    def add_relationship(
        self,
        *,
        head: SemanticEntity | None = None,
        tail: SemanticEntity | None = None,
        label: str | None = None,
        relationship: SemanticRelationship | None = None,
    ):
        if relationship is None:
            if head is None or tail is None or label is None:
                raise ValueError(
                    "Either a relationship or head, tail, and label must be provided"
                )
            relationship = SemanticRelationship(label, head, tail)
        else:
            assert isinstance(relationship.head, (str, SemanticEntity))
            assert isinstance(relationship.tail, (str, SemanticEntity))
        self.relationships.append(relationship)

    def resolve_relationships(self):
        for rel in self.relationships:
            if isinstance(rel.head, str):
                rel.head = self.entities[rel.head]
            if isinstance(rel.tail, str):
                rel.tail = self.entities[rel.tail]

    def resolve_literals(self):
        """
        Populate each entity’s `literals` dict from OWL data properties.
        """
        for dp in self.src_ontology.data_properties():
            prop_iri = dp.iri
            # dp[inst] returns the list of Python values for that individual
            for inst in dp.domain:  # or iterate dp.get_relations() if preferred
                values = dp[inst]
                for v in values:
                    # infer datatype IRI if declared; else default to xsd:string
                    datatype = dp.range[0].iri if dp.range else "xsd:string"
                    lit = Literal(v, datatype)
                    ent = self.entities.get(inst.iri)
                    if ent is not None:
                        ent.literals[prop_iri] = lit

    def add_literal_property(
        self, entity_uri: str, prop_uri: str, literal: Literal
    ) -> None:
        """
        Attach a new literal to an existing SemanticEntity.
        """
        ent = self.entities.get(entity_uri)
        if ent is None:
            raise KeyError(f"Entity '{entity_uri}' not in network")
        ent.literals[prop_uri] = literal

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
