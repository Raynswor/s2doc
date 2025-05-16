import enum

from .errors import LoadFromDictError


class TypographicLayout(enum.Enum):
    MIXED = 0
    ONE_COLUMN = 1
    TWO_COLUMN = 2
    THREE_COLUMN = 3
    FOUR_COLUMN = 4


class PageLayout:
    def __init__(
        self,
        rotation: float = 0,
        typography_style: TypographicLayout = TypographicLayout.MIXED,
        column_fractions: list[float] | None = None,
    ):
        self.rotation: float = rotation
        self.typography_style: TypographicLayout = typography_style
        self.column_fractions: list[float] = column_fractions or []

    def __repr__(self):
        if isinstance(self.typography_style, str):
            return f"<{self.rotation}°, {self.typography_style} at {self.column_fractions}>"
        else:
            return f"<{self.rotation}°, {self.typography_style.name} at {self.column_fractions}>"

    @classmethod
    def from_dict(cls, d: dict):
        try:
            return cls(
                d["rotation"],
                TypographicLayout[d["typography_style"].upper()],
                d["column_fractions"],
            )
        except KeyError as e:
            raise LoadFromDictError(cls.__name__, f"Missing key: {e}")

    def to_obj(self) -> dict:
        return {
            "rotation": self.rotation,
            "typography_style": self.typography_style
            if isinstance(self.typography_style, str)
            else self.typography_style.name,
            "column_fractions": self.column_fractions,
        }
