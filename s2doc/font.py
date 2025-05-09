from typing import Any

from .errors import LoadFromDictError


class Font:
    def __init__(self, label: str, color: str, size: float, style: Any = None) -> None:
        self.label = label
        self.color = color
        self.size = size
        self.style = style

    def is_super_script(self) -> bool:
        return "superscript" in self.style

    def is_sub_script(self) -> bool:
        return "subscript" in self.style

    @classmethod
    def from_dict(cls, d: dict):
        try:
            return cls(d["label"], d["color"], float(d["size"]), d["style"])
        except Exception as e:
            raise LoadFromDictError(cls.__name__, str(e))

    def to_obj(self) -> dict:
        return {
            "label": self.label,
            "color": self.color,
            "size": self.size,
            "style": self.style,
        }
