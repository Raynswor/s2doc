from .base import DocObj
from .element import Element
from .errors import LoadFromDictError
from .geometry import Region


class GroupedAreas(DocObj):
    def __init__(self, elements: list[Element]) -> None:
        super().__init__()
        self.elements: list[Element] = elements if elements else []

    @property
    def boundingBox(self) -> Region:
        return self.get_maximum_boundingBox()

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def get_maximum_boundingBox(self) -> Region:
        bb: tuple[float, float, float, float] = (10000, 10000, 0, 0)
        for vv in self.elements:
            bb = (
                min(vv.region.x1, bb[0]),
                min(vv.region.y1, bb[1]),
                max(vv.region.x2, bb[2]),
                max(vv.region.y2, bb[3]),
            )
        return Region(*bb, space=self.elements[0].region.space)

    def get_average_boundingBox(self) -> Region:
        bb: tuple[float, float, float, float] = (0, 0, 0, 0)
        for vv in self.elements:
            l = len(self.elements)
            bb = (
                (bb[0] + vv.region.x1) / l,
                (bb[1] + vv.region.y1) / l,
                (bb[2] + vv.region.x2) / l,
                (bb[3] + vv.region.y2) / l,
            )
        return Region(*bb, space=self.elements[0].region.space)

    @classmethod
    def from_dict(cls, d: dict) -> "GroupedAreas":
        try:
            elements = [Element.from_dict(element) for element in d["elements"]]
            return cls(elements)
        except KeyError as e:
            raise LoadFromDictError(cls.__name__, f"Missing key: {e}")

    def to_obj(self) -> dict:
        return {
            "elements": [x.to_obj() for x in self.elements],
        }

    def __repr__(self) -> str:
        return super().__repr__() + f"({len(self.elements)} elements)"
