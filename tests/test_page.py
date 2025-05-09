import pytest

from s2doc.errors import IncompatibleError
from s2doc.page import Page
from s2doc.pageLayout import PageLayout, TypographicLayout
from s2doc.space import Space


@pytest.fixture
def page():
    return Page(oid="page-1", number=1)


def test_page_initialization(page):
    assert page.oid == "page-1"
    assert page.number == 1
    assert isinstance(page.spaces, dict)
    assert "xml" in page.spaces
    assert "img" in page.spaces
    assert isinstance(page.layout, PageLayout)


def test_factor_between_spaces(page):
    page.spaces["xml"] = Space(label="xml", dimensions=[100.0, 200.0])
    page.spaces["img"] = Space(label="img", dimensions=[200.0, 400.0])
    factor = page.factor_between_spaces("xml", "img")
    assert factor == (2.0, 2.0)


def test_factor_between_spaces_incompatible(page):
    page.spaces["xml"] = Space(label="xml", dimensions=[100.0, 200.0])
    with pytest.raises(IncompatibleError):
        page.factor_between_spaces("xml", "nonexistent")


def test_set_typographic_columns(page):
    page.set_typographic_columns(columns=2)
    assert page.layout.typography_style ==  TypographicLayout.TWO_COLUMN

    page.set_typographic_columns(column_fractions=[0.5, 0.5])
    assert page.layout.column_fractions == [0.5, 0.5]


def test_set_typographic_columns_invalid(page):
    with pytest.raises(ValueError):
        page.set_typographic_columns(columns=4)

    with pytest.raises(ValueError):
        page.set_typographic_columns(column_fractions=[-0.1, 1.1])


def test_set_rotation(page):
    page.set_rotation(90)
    assert page.layout.rotation == 90

    with pytest.raises(ValueError):
        page.set_rotation(45)


def test_is_rotated(page):
    assert not page.is_rotated()
    page.set_rotation(90)
    assert page.is_rotated()


def test_repr(page):
    assert repr(page) == f"Page(page-1, 1, {page.layout})"


def test_from_dict():
    data = {
        "oid": "page-1",
        "number": 1,
        "img": None,
        "spaces": {
            "xml": {"label": "xml",
                    "dimensions": [100.0, 200.0],
                    "axis_directions": [True, False]
                    },
            "img": {"label": "img",
                    "dimensions": [200.0, 400.0],
                    "axis_directions": [True, False]
                    },
        },
        "layout": {},
    }
    page = Page.from_dict(data)
    assert page.oid == "page-1"
    assert page.number == 1
    assert page.spaces["xml"].dimensions == [100.0, 200.0]
    assert page.spaces["img"].dimensions == [200.0, 400.0]


def test_to_obj(page):
    page_dict = page.to_obj()
    assert page_dict["oid"] == "page-1"
    assert page_dict["number"] == 1
    assert "spaces" in page_dict
    assert "layout" in page_dict
