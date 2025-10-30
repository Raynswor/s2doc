from typing import Any

from .base import DocObj
from .errors import IncompatibleError, LoadFromDictError
from .geometry import Region


class Element(DocObj):
    def __init__(
        self,
        oid: str,
        category: str,
        region: Region,
        data: dict[str, Any] | None = None,
        confidence: float | None = None,
    ):
        self.oid: str = oid
        self.category: str = category
        self.region: Region = region
        self.data: dict[str, Any] = data if data else {}
        self.confidence: float | None = confidence

    @classmethod
    def create(
        cls,
        oid: str,
        category: str,
        region: Region,
        data: dict[str, Any] | None = None,
        confidence: float | None = None,
    ) -> "Element":
        """Factory method to instantiate the correct subclass."""
        # Lazy import to avoid circular imports and reduce module import cost.
        if category.lower() == "table":
            from .table import Table

            return Table(oid, region, data, confidence)
        if category.lower() == "table_cell":
            from .table import TableCell

            return TableCell(oid, region, data, confidence)
        return cls(oid, category, region, data, confidence)

    def __repr__(self) -> str:
        data_part = self.data if self.data else ""
        return f"<{self.oid}: {self.region} {data_part}>"

    def merge(
        self, other: "Element", merge_data: bool = True, merge_confidence: bool = True
    ):
        # check if the other element is of the same type
        if self.category.lower() != other.category.lower():
            raise IncompatibleError("type", self.category, other.__class__.__name__)
        # Update bounding region
        self.region = self.region.union(other.region)

        if merge_data and other.data:
            self_data = self.data
            # iterate once and use local variables for speed
            for k, v in other.data.items():
                if k in self_data:
                    existing = self_data[k]
                    # Merge when both items are the same container type, otherwise overwrite
                    if isinstance(existing, list) and isinstance(v, list):
                        existing.extend(v)
                    elif isinstance(existing, dict) and isinstance(v, dict):
                        existing.update(v)
                    elif isinstance(existing, str) and isinstance(v, str):
                        # string concatenation
                        self_data[k] = existing + v
                    else:
                        # Types differ or unsupported merge -> replace with other's value
                        self_data[k] = v
                else:
                    self_data[k] = v

        if merge_confidence:
            # Prefer the maximum if both present, otherwise take available value
            a = self.confidence
            b = other.confidence
            if a is None:
                self.confidence = b
            elif b is None:
                self.confidence = a
            else:
                self.confidence = a if a >= b else b

    @classmethod
    def from_dict(cls, d: dict) -> "Element":
        try:
            return cls.create(
                d["oid"],
                d["c"],
                Region.from_dict(d["r"]),
                data=d.get("data"),
                confidence=d.get("confidence"),
            )
        except KeyError as e:
            # Provide clearer error message about which key was missing
            missing = e.args[0] if e.args else str(e)
            raise LoadFromDictError(cls.__name__, f"missing key: {missing}, {e}")

    def to_obj(self) -> dict:
        dic: dict[str, Any] = {
            "oid": self.oid,
            "c": self.category,
            "r": self.region.to_obj(),
        }
        if self.data:
            dic["data"] = self.data
        if self.confidence is not None:
            dic["confidence"] = self.confidence
        return dic
