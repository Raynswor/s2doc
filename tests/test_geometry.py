import pytest

from src.errors import IncompatibleError, LoadFromDictError

from src.geometry import (
    LineRegion,
    PolylineRegion,
    RectangleRegion,
    Region,
    SpanRegion,
)


def test_region_intersects():
    region1 = RectangleRegion(0, 0, 10, 10)
    region2 = RectangleRegion(5, 5, 15, 15)
    region3 = RectangleRegion(20, 20, 30, 30)

    assert region1.intersects(region2) is True
    assert region1.intersects(region3) is False


def test_region_union():
    region1 = RectangleRegion(0, 0, 10, 10)
    region2 = RectangleRegion(5, 5, 15, 15)

    union_region = region1.union(region2)
    assert isinstance(union_region, RectangleRegion)
    assert union_region.x1 == 0
    assert union_region.y1 == 0
    assert union_region.x2 == 15
    assert union_region.y2 == 15


def test_region_intersection():
    region1 = RectangleRegion(0, 0, 10, 10)
    region2 = RectangleRegion(5, 5, 15, 15)
    region3 = RectangleRegion(20, 20, 30, 30)

    intersection_region = region1.intersection(region2)
    assert isinstance(intersection_region, RectangleRegion)
    assert intersection_region.x1 == 5
    assert intersection_region.y1 == 5
    assert intersection_region.x2 == 10
    assert intersection_region.y2 == 10

    assert region1.intersection(region3) is None


def test_region_contains():
    region1 = RectangleRegion(0, 0, 10, 10)
    region2 = RectangleRegion(2, 2, 8, 8)
    region3 = RectangleRegion(5, 5, 15, 15)

    assert region1.contains(region2) is True
    assert region1.contains(region3) is False


def test_region_distance():
    region1 = RectangleRegion(0, 0, 10, 10)
    region2 = RectangleRegion(20, 20, 30, 30)

    assert region1.distance(region2) > 0


def test_region_from_dict():
    rect_data = ["pr", 0, 0, 10, 10, "img"]
    span_data = ["s", 0, 10, "tokens"]

    rect_region = Region.from_dict(rect_data)
    span_region = Region.from_dict(span_data)

    assert isinstance(rect_region, RectangleRegion)
    assert rect_region.x1 == 0
    assert rect_region.y1 == 0
    assert rect_region.x2 == 10
    assert rect_region.y2 == 10

    assert isinstance(span_region, SpanRegion)
    assert span_region.start == 0
    assert span_region.end == 10


def test_region_from_dict_invalid():
    invalid_data = ["invalid", 0, 0, 10, 10, "img"]

    with pytest.raises(LoadFromDictError):
        Region.from_dict(invalid_data)


def test_region_convert_space():
    region = RectangleRegion(0, 0, 10, 10, "img")
    converted_region = region.convert_space([2, 2], "new_space")

    assert isinstance(converted_region, RectangleRegion)
    assert converted_region.x1 == 0
    assert converted_region.y1 == 0
    assert converted_region.x2 == 20
    assert converted_region.y2 == 20
    assert converted_region.space == "new_space"


def test_region_convert_space_same_space():
    region = RectangleRegion(0, 0, 10, 10, "img")
    converted_region = region.convert_space([2, 2], "img")

    assert converted_region == region


def test_region_intersects_incompatible_space():
    region1 = RectangleRegion(0, 0, 10, 10, "img")
    region2 = RectangleRegion(5, 5, 15, 15, "tokens")

    with pytest.raises(IncompatibleError):
        region1.intersects(region2)


def test_line_region():
    line_region = LineRegion(0, 0, 10, 10, "img")

    assert line_region.x1 == 0
    assert line_region.y1 == 0
    assert line_region.x2 == 10
    assert line_region.y2 == 10
    assert line_region.space == "img"
    assert line_region.get_points() == [(0, 0), (10, 10)]

    converted_line = line_region.convert_space([2, 2], "new_space")
    assert isinstance(converted_line, LineRegion)
    assert converted_line.x1 == 0
    assert converted_line.y1 == 0
    assert converted_line.x2 == 20
    assert converted_line.y2 == 20
    assert converted_line.space == "new_space"


def test_polyline_region():
    points = [(0, 0), (5, 5), (10, 10)]
    polyline_region = PolylineRegion(points, "img")

    assert polyline_region.points == points
    assert polyline_region.space == "img"
    assert polyline_region.get_points() == points

    converted_polyline = polyline_region.convert_space([2, 2], "new_space")
    assert isinstance(converted_polyline, PolylineRegion)
    assert converted_polyline.points == [(0, 0), (10, 10), (20, 20)]
    assert converted_polyline.space == "new_space"
