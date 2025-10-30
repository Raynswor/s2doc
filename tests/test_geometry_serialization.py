from src.s2doc.geometry import PolygonRegion, RectangleRegion, Region, SpanRegion


def test_rectangle_roundtrip():
    r = RectangleRegion(0, 0, 10, 10, "img")
    obj = r.to_obj()
    rr = Region.from_dict(obj)
    assert isinstance(rr, RectangleRegion)
    assert rr.x1 == 0
    assert rr.y1 == 0
    assert rr.x2 == 10
    assert rr.y2 == 10


def test_polygon_roundtrip():
    pts = [(0, 0), (0, 10), (10, 10), (10, 0)]
    p = PolygonRegion(pts, "img")
    obj = p.to_obj()
    pr = Region.from_dict(obj)
    assert isinstance(pr, PolygonRegion)
    assert pr.points == pts


def test_span_roundtrip():
    s = SpanRegion(0, 10, "tokens")
    obj = s.to_obj()
    sr = Region.from_dict(obj)
    assert isinstance(sr, SpanRegion)
    assert sr.start == 0
    assert sr.end == 10
