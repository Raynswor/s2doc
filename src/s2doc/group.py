import numpy as np

from .base import DocObj
from .element import Element
from .errors import LoadFromDictError
from .geometry import RectangleRegion


class GroupedElements(DocObj):
    def __init__(self, elements: list[Element]) -> None:
        super().__init__()
        self.elements: list[Element] = elements if elements else []

    @property
    def region(self) -> RectangleRegion:
        return self.get_maximum_region()

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def get_maximum_region(self) -> RectangleRegion:
        n = len(self.elements)
        if n == 0:
            return RectangleRegion(0, 0, 0, 0)

        arr = np.array(
            [vv.region.bounds for vv in self.elements], dtype=float
        )  # shape (n,4)
        x1_min = float(arr[:, 0].min())
        y1_min = float(arr[:, 1].min())
        x2_max = float(arr[:, 2].max())
        y2_max = float(arr[:, 3].max())
        return RectangleRegion(
            x1_min, y1_min, x2_max, y2_max, space=self.elements[0].region.space
        )

    def get_average_region(self) -> RectangleRegion:
        n = len(self.elements)
        if n == 0:
            return RectangleRegion(0, 0, 0, 0)

        arr = np.array(
            [vv.region.bounds for vv in self.elements], dtype=float
        )  # shape (n,4)
        x1, y1, x2, y2 = arr.mean(axis=0).tolist()
        return RectangleRegion(
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            space=self.elements[0].region.space,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "GroupedElements":
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
