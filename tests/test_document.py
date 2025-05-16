import pytest

from src.document import Document
from src.element import Element
from src.errors import AreaNotFoundError, PageNotFoundError
from src.geometry import RectangleRegion
from src.page import Page
from src.semantics import SemanticEntity, SemanticType
from src.space import Space


@pytest.fixture
def document():
    return Document(oid="test_doc")


def test_add_page(document):
    page_id = document.add_page()
    assert page_id in document.pages


def test_add_element(document):
    page_id = document.add_page(Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, False]),
        } ))
    region = RectangleRegion(0, 0, 100, 100, "img")
    element_id = document.add_element(page=page_id, category="Test", region=region)
    assert element_id in document.elements


def test_add_element_invalid_page(document):
    region = RectangleRegion(0, 0, 100, 100, "img")
    with pytest.raises(PageNotFoundError):
        document.add_element(page="invalid_page", category="Test", region=region)


def test_delete_element(document):
    page_id = document.add_page(Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, False]),
        } ))
    region = RectangleRegion(0, 0, 100, 100, "img")
    element_id = document.add_element(page=page_id, category="Test", region=region)
    document.delete_element(element_id)
    assert element_id not in document.elements


def test_replace_element(document):
    page_id = document.add_page(Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, False]),
        } ))
    region = RectangleRegion(0, 0, 100, 100, "img")
    old_element_id = document.add_element(page=page_id, category="Test", region=region)
    new_element = Element.create("new_element", region=region, category="Test")
    document.replace_element(old_element_id, new_element)
    assert "new_element" in document.elements
    assert old_element_id not in document.elements


def test_get_visible_elements_ydown(document):
    page_id = document.add_page(
        Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, False]),
        } )
    )
    region_1 = RectangleRegion(0, 0, 100, 100, "img")
    region_2 = RectangleRegion(150, 0, 250, 100, "img")
    region_3 = RectangleRegion(300, 0, 350, 100, "img")
    region_4 = RectangleRegion(50, 150, 200, 250, "img")
    region_5 = RectangleRegion(150, 110, 250, 120, "img")

    element_id_1 = document.add_element(page=page_id, category="Test", region=region_1)
    element_id_2 = document.add_element(page=page_id, category="Test", region=region_2)
    element_id_3 = document.add_element(page=page_id, category="Test", region=region_3)
    element_id_4 = document.add_element(page=page_id, category="Test", region=region_4)
    element_id_5 = document.add_element(page=page_id, category="Test", region=region_5)

    # | 1 |  | 2 |  | 3 |
    #        | 5 |
    #   |  4  |
    #
    #

    visible_elements = [x.oid for x in document.get_visible_elements(element_id_2,  "right", "Test")]
    # should be element3
    assert len(visible_elements) == 1
    assert element_id_3 in visible_elements

    visible_elements = [x.oid for x in document.get_visible_elements(element_id_2, "left", "Test")]
    assert len(visible_elements) == 1
    assert element_id_1 in visible_elements

    visible_elements = [x.oid for x in document.get_visible_elements(element_id_4, "above", "Test")]
    assert len(visible_elements) == 2
    assert element_id_1 in visible_elements
    assert element_id_5 in visible_elements

    visible_elements = [x.oid for x in document.get_visible_elements(element_id_2, "below", "Test")]
    assert len(visible_elements) == 1
    assert element_id_5 in visible_elements


def test_get_visible_elements_yup(document):
    page_id = document.add_page(
        Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, True]),
        } )
    )
    region_1 = RectangleRegion(0, 0, 100, 100, "img")
    region_2 = RectangleRegion(150, 0, 250, 100, "img")
    region_3 = RectangleRegion(300, 0, 350, 100, "img")
    region_4 = RectangleRegion(50, 150, 200, 250, "img")
    region_5 = RectangleRegion(150, 110, 250, 120, "img")

    element_id_1 = document.add_element(page=page_id, category="Test", region=region_1)
    element_id_2 = document.add_element(page=page_id, category="Test", region=region_2)
    element_id_3 = document.add_element(page=page_id, category="Test", region=region_3)
    element_id_4 = document.add_element(page=page_id, category="Test", region=region_4)
    element_id_5 = document.add_element(page=page_id, category="Test", region=region_5)

    #   |  4  |
    #        | 5 |
    # | 1 |  | 2 |  | 3 |

    visible_elements = [x.oid for x in document.get_visible_elements(element_id_2,  "right", "Test")]
    # should be element3
    assert len(visible_elements) == 1
    assert element_id_3 in visible_elements

    visible_elements = [x.oid for x in document.get_visible_elements(element_id_2, "left", "Test")]
    assert len(visible_elements) == 1
    assert element_id_1 in visible_elements

    visible_elements = [x.oid for x in document.get_visible_elements(element_id_5, "above", "Test")]
    assert len(visible_elements) == 1
    assert element_id_4 in visible_elements

    visible_elements = [x.oid for x in document.get_visible_elements(element_id_4, "below", "Test")]
    assert len(visible_elements) == 2
    assert element_id_1 in visible_elements
    assert element_id_5 in visible_elements



def test_get_element_data_value(document):
    page_id = document.add_page(
        Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, True]),
        } )
    )
    region = RectangleRegion(0, 0, 100, 100, "img")
    element_id = document.add_element(
        page=page_id, category="Test", region=region, data={"content": "test_content"}
    )
    data_value = document.get_element_data_value(element_id)
    assert data_value == ["test_content"]


def test_add_revision(document):
    document.add_revision("New Revision")
    assert len(document.revisions) == 2
    assert document.revisions[-1].comment == "New Revision"


def test_get_element_type(document):
    page_id = document.add_page(
        Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, True]),
        } )
    )
    region = RectangleRegion(0, 0, 100, 100, "img")
    document.add_element(page=page_id, category="Test", region=region)
    elements = list(document.get_element_type("Test"))
    assert len(elements) == 1
    assert elements[0].category == "Test"


def test_is_referenced_by(document):
    page_id = document.add_page(
        Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, True]),
        } )
    )
    region = RectangleRegion(0, 0, 100, 100, "img")
    element_id_1 = document.add_element(page=page_id, category="Test", region=region)
    element_id_2 = document.add_element(
        page=page_id, category="Test", region=region, referenced_by=element_id_1
    )
    assert document.is_referenced_by(element_id_2, element_id_1)


def test_get_img_snippet_invalid_element(document):
    with pytest.raises(AreaNotFoundError):
        document.get_img_snippet("invalid_element")


def test_delete_elements(document):
    page_id = document.add_page(
        Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, True]),
        } )
    )
    region_1 = RectangleRegion(0, 0, 100, 100, "img")
    region_2 = RectangleRegion(150, 0, 250, 100, "img")
    element_id_1 = document.add_element(page=page_id, category="Test", region=region_1)
    element_id_2 = document.add_element(page=page_id, category="Test", region=region_2)

    document.delete_elements([element_id_1, element_id_2])
    assert element_id_1 not in document.elements
    assert element_id_2 not in document.elements


def test_find_page_of_element(document):
    page_id = document.add_page(
        Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, True]),
        } )
    )
    region = RectangleRegion(0, 0, 100, 100, "img")
    element_id = document.add_element(page=page_id, category="Test", region=region)

    found_page_id = document.find_page_of_element(element_id)
    assert found_page_id == page_id


def test_get_latest_elements(document):
    page_id = document.add_page(
        Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, True]),
        } )
    )
    region = RectangleRegion(0, 0, 100, 100, "img")
    element_id = document.add_element(page=page_id, category="Test", region=region)

    latest_elements = document.get_latest_elements()
    assert element_id in latest_elements


def test_get_elements_at_position(document):
    page_id = document.add_page(
        Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, True]),
        })
    )
    region = RectangleRegion(0, 0, 100, 100, "img")
    element_id = document.add_element(page=page_id, category="Test", region=region)

    elements = document.get_elements_at_position(page_id, 50, 50, "Test")
    assert len(elements) == 1
    assert elements[0].oid == element_id


def test_get_img_snippets(document):
    page_id = document.add_page(
        Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, True]),
        } )
    )
    region = RectangleRegion(0, 0, 100, 100, "img")
    element_id = document.add_element(page=page_id, category="Test", region=region)

    snippets = list(document.get_img_snippets([document.get_element_obj(element_id)]))
    assert len(snippets) == 1
    assert snippets[0].shape[0] > 0
    assert snippets[0].shape[1] > 0


def test_get_img_snippet_from_bb(document):
    page_id = document.add_page(
        Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, True]),
        } )
    )
    region = RectangleRegion(0, 0, 100, 100, "img")

    snippet = document.get_img_snippet_from_bb(region, page_id, as_string=False)
    assert snippet is not None
    assert snippet.size[0] > 0
    assert snippet.size[1] > 0


def test_annotate_element_with_uri(document):
    page_id = document.add_page(
        Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, True]),
        } )
    )
    region = RectangleRegion(0, 0, 100, 100, "img")
    element_id = document.add_element(page=page_id, category="Test", region=region)

    uri = "http://example.com/entity"
    document.annotate_element_with_uri(element_id, uri)
    assert uri in document.semantic_references.get_descendants(element_id)


def test_annotate_element_with_entity(document):
    page_id = document.add_page(
        Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, True]),
        } )
    )
    region = RectangleRegion(0, 0, 100, 100, "img")
    element_id = document.add_element(page=page_id, category="Test", region=region)

    entity = SemanticEntity(uri="http://example.com/entity")

    document.semantic_network.add_entity(entity)
    document.annotate_element_with_entity(element_id, entity)

    assert entity.uri in document.semantic_references.get_descendants(element_id)


def test_annotate_element_with_type(document):
    page_id = document.add_page(
        Page("page-0", 0, None, {
            "img": Space("img", [1000,1000], [True, True]),
        })
    )
    region = RectangleRegion(0, 0, 100, 100, "img")
    element_id = document.add_element(page=page_id, category="Test", region=region)

    semantic_type = SemanticType(uri="http://example.com/type")
    document.semantic_network.add_type(semantic_type)

    document.annotate_element_with_type(element_id, semantic_type)
    assert semantic_type.uri in document.semantic_references.get_descendants(element_id)


def test_document_to_obj_and_from_dict(document):
    import json
    # Convert the document to a dictionary
    document_json = document.to_json()

    # Deserialize the JSON back to a dictionary
    loaded_dict = json.loads(document_json)

    # Create a new Document instance from the dictionary
    loaded_document = Document.from_dict(loaded_dict)

    # Assert that the original and loaded documents have the same attributes
    assert document.oid == loaded_document.oid
    assert document.pages.to_obj() == loaded_document.pages.to_obj()
    assert document.elements.to_obj() == loaded_document.elements.to_obj()
    assert document.references.to_obj() == loaded_document.references.to_obj()
    assert [rev.to_obj() for rev in document.revisions] == [
        rev.to_obj() for rev in loaded_document.revisions
    ]
    assert [font.to_obj() for font in document.fonts] == [
        font.to_obj() for font in loaded_document.fonts
    ]
    assert document.metadata == loaded_document.metadata
    assert document.raw_pdf == loaded_document.raw_pdf
    assert document.semantic_network.to_obj() == loaded_document.semantic_network.to_obj()
    assert document.semantic_references.to_obj() == loaded_document.semantic_references.to_obj()