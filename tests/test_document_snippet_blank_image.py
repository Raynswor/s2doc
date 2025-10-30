from PIL import Image

from src.s2doc.document import Document
from src.s2doc.geometry import RectangleRegion
from src.s2doc.page import Page


def test_get_img_snippet_creates_blank():
    doc = Document("d1")
    page = Page("pg1", 1, img=None)
    # give the page an img space with concrete dimensions
    page.spaces["img"].dimensions = [100, 100]
    doc.pages.add(page)

    region = RectangleRegion(0, 0, 50, 50, "img")
    img = doc.get_img_snippet_from_bb(region, page, as_string=False)
    assert isinstance(img, Image.Image)
    assert img.size == (50, 50)
