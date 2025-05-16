from collections.abc import Set

from .base import DocObj
from .errors import LoadFromDictError


class Revision(DocObj):
    def __init__(
        self,
        timestamp: str,
        objects: Set[str] | None = None,
        comment: str = "",
        del_objs: Set[str] | None = None,
        reference_revoked: bool = False,
    ):
        self.timestamp: str = timestamp
        self.objects: Set[str] = set(objects) if objects else set()
        self.comment: str = comment
        self.del_objs: Set[str] = set(del_objs) if del_objs else set()
        self.reference_revoked: bool = reference_revoked

    def adjust_objs(self, other: Set[str]) -> Set[str]:
        return set(self.objects) | other - set(self.del_objs)

    @classmethod
    def from_dict(cls, d: dict) -> "Revision":
        try:
            return Revision(
                d["timestamp"],
                d.get("objects", set()),
                d.get("comment", ""),
                d.get("del_objs", None),
                d.get("reference_revoked", False),
            )
        except Exception as e:
            raise LoadFromDictError(cls.__name__, str(e))

    def to_obj(self) -> dict:
        d = {
            "timestamp": self.timestamp,
            "objects": self.objects,
            "comment": self.comment,
            "reference_revoked": self.reference_revoked,
        }
        if self.del_objs:
            d["del_objs"] = self.del_objs
        return d
