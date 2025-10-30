from .errors import LoadFromDictError


class Space:
    __slots__ = ("label", "dimensions", "axis_directions")

    def __init__(
        self, label: str, dimensions: list[float], axis_directions: list[bool] = []
    ) -> None:
        """
        Initialize a Space object with a label, dimensions, and optional axis directions.

        Args:
            label: The label for the space.
            dimensions: A list of dimensions for the space.
            axis_directions: A list of booleans indicating the direction of each axis. True indicates the natural direction,
                and False indicates the opposite direction. If not provided, all axes are assumed to be in the natural direction.
        """
        self.label = label
        self.dimensions = dimensions
        if len(axis_directions) == 0:
            self.axis_directions = [True] * len(dimensions)
        elif len(axis_directions) != len(dimensions):
            raise ValueError("axis_directions must match the length of dimensions")
        else:
            for i in range(len(axis_directions)):
                if axis_directions[i] not in [True, False]:
                    raise ValueError("axis_directions must be a list of booleans")
            self.axis_directions = axis_directions

    @property
    def width(self) -> float:
        return self.dimensions[0] if len(self.dimensions) > 0 else 0.0

    @property
    def height(self) -> float:
        return self.dimensions[1] if len(self.dimensions) > 1 else 0.0

    @property
    def depth(self) -> float:
        return self.dimensions[2] if len(self.dimensions) > 2 else 0.0

    def to_obj(self) -> dict[str, str | list[float]]:
        return {
            "label": self.label,
            "dimensions": self.dimensions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Space":
        try:
            return cls(
                label=d["label"],
                dimensions=[float(x) for x in d["dimensions"]],
                axis_directions=d.get("axis_directions", []),
            )
        except Exception as e:
            raise LoadFromDictError(cls.__name__, str(e))
