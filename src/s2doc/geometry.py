from abc import ABC, abstractmethod

from shapely import LineString, Polygon

from .errors import IncompatibleError, LoadFromDictError


def check_space(func):
    def wrapper(self, other: "Region", *args, **kwargs):
        if not isinstance(other, Region):
            raise IncompatibleError("type", "Region", str(type(other)))
        if self.space != other.space:
            raise IncompatibleError("space", self.space, other.space)
        return func(self, other, *args, **kwargs)

    return wrapper


class Region(ABC):
    def __init__(self, shape: Polygon | LineString, space: str = "img") -> None:
        self._shape = shape
        self._space = space

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self._shape.bounds

    @property
    def space(self) -> str:
        return self._space

    @abstractmethod
    def to_obj(self) -> list:
        pass

    @abstractmethod
    def get_points(self) -> list[tuple[float, float]]:
        pass

    @check_space
    def intersects(self, other: "Region") -> bool:
        return self._shape.intersects(other._shape)

    @check_space
    def union(self, other: "Region") -> "Region":
        union_shape = self._shape.union(other._shape)
        if isinstance(union_shape, Polygon):
            return RectangleRegion(*union_shape.bounds, self._space)
        elif isinstance(union_shape, LineString):
            return SpanRegion(
                int(union_shape.bounds[0]), int(union_shape.bounds[2]), self._space
            )
        else:
            raise ValueError("Union resulted in an unsupported geometry type")

    @check_space
    def intersection(self, other: "Region") -> "Region | None":
        inter = self._shape.intersection(other._shape)
        if inter.is_empty:
            return None
        if isinstance(inter, Polygon):
            return RectangleRegion(*inter.bounds, self._space)
        elif isinstance(inter, LineString):
            return SpanRegion(int(inter.bounds[0]), int(inter.bounds[2]), self._space)
        else:
            raise ValueError("Intersection resulted in an unsupported geometry type")

    @check_space
    def contains(self, other: "Region") -> bool:
        return self._shape.contains(other._shape)

    @check_space
    def distance(self, other: "Region") -> float:
        return self._shape.distance(other._shape)

    @classmethod
    def from_dict(cls, d: list) -> "Region":
        match d[0]:
            case "s":
                return SpanRegion.from_dict(d[1:])
            case "pr":
                return RectangleRegion.from_dict(d[1:])
            case "l":
                return LineRegion.from_dict(d[1:])
            case "pl":
                return PolylineRegion.from_dict(d[1:])
            case _:
                raise LoadFromDictError(
                    cls.__class__.__name__, "Incorrect type for parameters"
                )

    @abstractmethod
    def convert_space(self, factors: list[float], space: str) -> "Region":
        pass

    @abstractmethod
    def __eq__(self, value: object) -> bool:
        pass


class SpanRegion(Region):
    """
    One-dimensional region represented by a start and end point.
    """

    def __init__(self, start: int, end: int, space: str = "tokens") -> None:
        super().__init__(LineString([(start, 0), (end, 0)]), space)
        self.start = start
        self.end = end

    def to_obj(self) -> list:
        return ["s", self.start, self.end, self.space]

    def get_points(self) -> list[tuple[float, float]]:
        return [(self.start, self.end)]

    @classmethod
    def from_dict(cls, d: list) -> "SpanRegion":
        if len(d) != 3:
            raise LoadFromDictError(cls.__name__, "Incorrect number of parameters")
        return cls(d[0], d[1], d[2])

    def convert_space(self, factors: list[float], space: str) -> "SpanRegion":
        if self.space == space:
            return self
        else:
            raise IncompatibleError("space", self.space, space)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SpanRegion):
            return NotImplemented
        return (
            self.space == value.space
            and self.start == value.start
            and self.end == value.end
        )


class RectangleRegion(Region):
    """
    Two-dimensional region represented by a bounding box.
    """

    def __init__(
        self, x1: float, y1: float, x2: float, y2: float, space: str = "img"
    ) -> None:
        super().__init__(Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)]), space)

    @property
    def x1(self) -> float:
        return self.bounds[0]

    @property
    def y1(self) -> float:
        return self.bounds[1]

    @property
    def x2(self) -> float:
        return self.bounds[2]

    @property
    def y2(self) -> float:
        return self.bounds[3]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def middle(self) -> tuple[float, float]:
        x, y = self._shape.centroid.coords[0]
        return float(x), float(y)

    def is_vertical(self) -> bool:
        return self.height > self.width

    def is_horizontal(self) -> bool:
        return self.width > self.height

    def to_obj(self) -> list:
        return ["pr", self.x1, self.y1, self.x2, self.y2, self.space]

    def get_points(self) -> list[tuple[float, float]]:
        return [(self.x1, self.y1), (self.x2, self.y2)]

    @classmethod
    def from_dict(cls, d: list) -> "RectangleRegion":
        if len(d) != 5:
            raise LoadFromDictError(cls.__name__, "Incorrect number of parameters")
        return cls(d[0], d[1], d[2], d[3], d[4])

    def convert_space(self, factors: list[float], space: str) -> "RectangleRegion":
        if self.space == space:
            return self
        return RectangleRegion(
            x1=self.x1 * factors[0],
            y1=self.y1 * factors[1],
            x2=self.x2 * factors[0],
            y2=self.y2 * factors[1],
            space=space,
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, RectangleRegion) and not isinstance(value, PolygonRegion) and not isinstance(value, LineRegion):
            return NotImplemented
        elif isinstance(value, PolygonRegion):
            return self._shape.equals(value._shape)
        elif isinstance(value, LineRegion):
            return self._shape.equals(value._shape)
        return (
            self.space == value.space
            and self.x1 == value.x1
            and self.y1 == value.y1
            and self.x2 == value.x2
            and self.y2 == value.y2
        )


class PolygonRegion(Region):
    """
    Two-dimensional region represented by a polygon.
    """

    def __init__(self, points: list[tuple[float, float]], space: str = "img") -> None:
        super().__init__(Polygon(points), space)
        self.points = points

    def to_obj(self) -> list:
        return ["pr", self.points, self.space]

    def get_points(self) -> list[tuple[float, float]]:
        return self.points

    @classmethod
    def from_dict(cls, d: list) -> "PolygonRegion":
        if len(d) != 3:
            raise LoadFromDictError(cls.__name__, "Incorrect number of parameters")
        return cls(d[0], d[1])

    def convert_space(self, factors: list[float], space: str) -> "PolygonRegion":
        if self.space == space:
            return self
        return PolygonRegion(
            points=[(x * factors[0], y * factors[1]) for x, y in self.points],
            space=space,
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, PolygonRegion) and not isinstance(value, RectangleRegion) and not isinstance(value, LineRegion):
            return NotImplemented
        elif isinstance(value, RectangleRegion):
            return self._shape.equals(value._shape)
        elif isinstance(value, LineRegion):
            return self._shape.equals(value._shape)
        return (
            self.space == value.space
            and self.points == value.points
            and self._shape.equals(value._shape)
        )


class LineRegion(Region):
    """
    One-dimension region in a 2D space represented by a straight line.
    """

    def __init__(
        self, x1: float, y1: float, x2: float, y2: float, space: str = "img"
    ) -> None:
        super().__init__(LineString([(x1, y1), (x2, y2)]), space)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def to_obj(self) -> list:
        return ["l", self.x1, self.y1, self.x2, self.y2, self.space]

    def get_points(self) -> list[tuple[float, float]]:
        return [(self.x1, self.y1), (self.x2, self.y2)]

    @classmethod
    def from_dict(cls, d: list) -> "LineRegion":
        if len(d) != 5:
            raise LoadFromDictError(cls.__name__, "Incorrect number of parameters")
        return cls(d[0], d[1], d[2], d[3], d[4])

    def convert_space(self, factors: list[float], space: str) -> "LineRegion":
        if self.space == space:
            return self
        return LineRegion(
            x1=self.x1 * factors[0],
            y1=self.y1 * factors[1],
            x2=self.x2 * factors[0],
            y2=self.y2 * factors[1],
            space=space,
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, LineRegion) and not isinstance(value, RectangleRegion) and not isinstance(value, PolygonRegion):
            return NotImplemented
        elif isinstance(value, RectangleRegion):
            return self._shape.equals(value._shape)
        elif isinstance(value, PolygonRegion):
            return self._shape.equals(value._shape)
        return (
            self.space == value.space
            and self.x1 == value.x1
            and self.y1 == value.y1
            and self.x2 == value.x2
            and self.y2 == value.y2
        )


class PolylineRegion(Region):
    """
    Two-dimensional region represented by a polyline.
    """

    def __init__(self, points: list[tuple[float, float]], space: str = "img") -> None:
        super().__init__(LineString(points), space)
        self.points = points

    def to_obj(self) -> list:
        return ["pl", self.points, self.space]

    def get_points(self) -> list[tuple[float, float]]:
        return self.points

    @classmethod
    def from_dict(cls, d: list) -> "PolylineRegion":
        if len(d) != 3:
            raise LoadFromDictError(cls.__name__, "Incorrect number of parameters")
        return cls(d[0], d[1])

    def convert_space(self, factors: list[float], space: str) -> "PolylineRegion":
        if self.space == space:
            return self
        return PolylineRegion(
            points=[(x * factors[0], y * factors[1]) for x, y in self.points],
            space=space,
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, PolylineRegion) and not isinstance(value, PolygonRegion) and not isinstance(value, RectangleRegion) and not isinstance(value, LineRegion):
            return NotImplemented
        elif isinstance(value, RectangleRegion):
            return self._shape.equals(value._shape)
        elif isinstance(value, PolygonRegion):
            return self._shape.equals(value._shape)
        elif isinstance(value, LineRegion):
            return self._shape.equals(value._shape)
        return (
            self.space == value.space
            and self.points == value.points
            and self._shape.equals(value._shape)
        )
