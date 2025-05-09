import re
import pytest

from s2doc.element import Element, Table
from s2doc.errors import IncompatibleError
from s2doc.geometry import RectangleRegion


def test_element_initialization():
    region = RectangleRegion(0, 0, 10, 10)
    element = Element("e1", "text", region, {"key": "value"}, 0.9)

    assert element.oid == "e1"
    assert element.category == "text"
    assert element.region == region
    assert element.data == {"key": "value"}
    assert element.confidence == 0.9


def test_element_create():
    region = RectangleRegion(0, 0, 10, 10)
    element = Element.create("e1", "text", region)
    assert isinstance(element, Element)
    assert element.category == "text"

    table = Element.create("t1", "table", region)
    assert isinstance(table, Table)
    assert table.category == "Table"


def test_element_merge():
    region1 = RectangleRegion(0, 0, 10, 10)
    region2 = RectangleRegion(5, 5, 15, 15)
    element1 = Element("e1", "text", region1, {"key1": "value1"}, 0.8)
    element2 = Element("e2", "text", region2, {"key2": "value2"}, 0.9)

    element1.merge(element2)

    assert element1.region == region1.union(region2)
    assert element1.data == {"key1": "value1", "key2": "value2"}
    assert element1.confidence == 0.9


def test_element_merge_incompatible():
    region = RectangleRegion(0, 0, 10, 10)
    element1 = Element("e1", "text", region)
    table = Table("t1", region)

    with pytest.raises(IncompatibleError, match=re.escape("(type): text -> Table")):
        element1.merge(table)


def test_element_from_dict():
    data = {
        "oid": "e1",
        "c": "text",
        "r": ["pr", 0, 0, 10, 10, "img"],
        "data": {"key": "value"},
        "confidence": 0.9,
    }
    element = Element.from_dict(data)

    assert element.oid == "e1"
    assert element.category == "text"
    assert element.region == RectangleRegion(0, 0, 10, 10)
    assert element.data == {"key": "value"}
    assert element.confidence == 0.9


def test_element_to_obj():
    region = RectangleRegion(0, 0, 10, 10)
    element = Element("e1", "text", region, {"key": "value"}, 0.9)
    obj = element.to_obj()

    assert obj == {
        "oid": "e1",
        "c": "text",
        "r": ["pr", 0, 0, 10, 10, "img"],
        "data": {"key": "value"},
        "confidence": 0.9,
    }
