from .base import DocObj
from .errors import IncompatibleError, LoadFromDictError
from .pageLayout import PageLayout, TypographicLayout
from .space import Space


class Page(DocObj):
    """
    Represents a page in a document with its properties and layout information.

    Attributes:
        oid: Unique identifier for the page
        number: Page number in the document
        img: Image data or path
        spaces: Dimensions of the page in generic coordinate space
        layout: Page layout information
    """

    def __init__(
        self,
        oid: str,
        number: int,
        img: str | None = None,
        spaces: dict[str, Space] | None = None,
        layout: PageLayout | None = None,
        metadata: dict[str, str] | None = None,
    ):
        self.oid: str = oid
        self.number: int = number
        self.img: str | None = img
        self.spaces: dict[str, Space] = spaces or {
            "xml": Space(
                label="xml", dimensions=[0.0, 0.0], axis_directions=[True, False]
            ),
            "img": Space(
                label="img", dimensions=[0.0, 0.0], axis_directions=[True, False]
            ),
        }
        self.layout: PageLayout = layout or PageLayout()
        self._cache_space_conversion: dict[tuple[str, str], list[float]] = {}
        self.metadata: dict[str, str] = metadata or {}

    def factor_between_spaces(self, from_space: str, to_space: str) -> list[float]:
        key = (from_space, to_space)
        if key in self._cache_space_conversion:
            return self._cache_space_conversion[key]
        if from_space not in self.spaces or to_space not in self.spaces:
            raise IncompatibleError("space", from_space, to_space)

        from_dim = self.spaces[from_space]
        to_dim = self.spaces[to_space]

        if len(from_dim.dimensions) != len(to_dim.dimensions):
            raise IncompatibleError("space", from_space, to_space)

        # TODO: currently only 2D spaces are supported
        conversion_factor = [
            to_dim.dimensions[0] / from_dim.dimensions[0],
            to_dim.dimensions[1] / from_dim.dimensions[1],
        ]
        self._cache_space_conversion[key] = conversion_factor
        return conversion_factor

    def set_typographic_columns(
        self, columns: int | None = None, column_fractions: list[float] | None = None
    ) -> None:
        """
        Set the number of typographic columns and their fractions.
        Args:
            columns: Number of typographic columns (1, 2, or 3)
            column_fractions: list of fractions for each column
        """
        if columns is not None:
            if columns not in {1, 2, 3}:
                raise ValueError(f"Number of columns must be 1, 2, or 3, not {columns}")
            self.layout.typography_style = TypographicLayout(columns)
        if column_fractions is not None:
            if any(fraction < 0 or fraction > 1 for fraction in column_fractions):
                raise ValueError("All column fractions must be between 0 and 1.")
            self.layout.column_fractions = column_fractions
            if self.layout.typography_style == TypographicLayout.MIXED:
                self.layout.typography_style = TypographicLayout(len(column_fractions))

    def set_rotation(self, rotation: int) -> None:
        """
        Set page rotation in degrees.

        Args:
            rotation: Rotation angle in degrees (must be 0, 90, 180, or 270)
        """
        if rotation not in {0, 90, 180, 270}:
            raise ValueError(f"Rotation must be 0, 90, 180, or 270, not {rotation}")
        self.layout.rotation = rotation

    def is_rotated(self):
        return self.layout.rotation != 0

    def __repr__(self):
        return f"Page({self.oid}, {self.number}, {self.layout})"

    @classmethod
    def from_dict(cls, d: dict) -> "Page":
        layout = (
            PageLayout.from_dict(d["layout"]) if "layout" in d and d["layout"] else None
        )

        if "xml_width" in d:
            dimensions = {
                "xml": Space(
                    label="xml",
                    dimensions=[float(d["xml_width"]), float(d["xml_height"])],
                    axis_directions=[True, False],
                ),
                "img": Space(
                    label="img",
                    dimensions=[float(d["img_width"]), float(d["img_height"])],
                    axis_directions=[True, False],
                ),
            }
        else:
            dimensions = {k: Space.from_dict(v) for k, v in d["spaces"].items()}

        try:
            return cls(
                oid=d["oid"],
                number=d["number"],
                img=d["img"],
                spaces=dimensions,
                layout=layout,
                metadata=d.get("metadata", {}),
            )
        except Exception as e:
            raise LoadFromDictError(cls.__name__, str(e))

    def to_obj(self) -> dict:
        return {
            "oid": self.oid,
            "number": self.number,
            "img": self.img,
            "spaces": {k: v.to_obj() for k, v in self.spaces.items()},
            "layout": self.layout.to_obj(),
            "metadata": self.metadata,
        }
