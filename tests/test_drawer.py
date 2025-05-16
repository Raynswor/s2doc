import pytest

from src import (
    Document,
    Drawer,
    Page,
    ReferenceGraph,
    SemanticEntity,
    Space,
)
from src.geometry import RectangleRegion
from src.semantics import SemanticType


@pytest.fixture
def sample_graph():
    graph = ReferenceGraph()
    graph.add_reference("A", "B")
    graph.add_reference("B", "C")
    graph.add_reference("A", "D")
    return graph


@pytest.fixture
def sample_document():
    document = Document(
        oid="DocA",
    )
    page = document.add_page(
        Page(
            "page-0",
            0,
            None,
            {
                "img": Space("img", [1000, 1000], [True, False]),
            },
        )
    )
    document.add_element(
        page, "Table", region=RectangleRegion(0, 0, 0, 0, "img"), element_id="Table1"
    )
    document.add_element(
        page,
        "Cell",
        region=RectangleRegion(0, 0, 0, 0, "img"),
        element_id="Cell2",
        referenced_by=["Table1"],
    )
    document.add_element(
        page,
        "Row",
        region=RectangleRegion(0, 0, 0, 0, "img"),
        element_id="Row1",
        referenced_by=["Table1"],
        references=["Cell2"],
    )
    document.add_element(
        page,
        "Column",
        region=RectangleRegion(0, 0, 0, 0, "img"),
        element_id="Column1",
        referenced_by=["Table1"],
    )
    document.add_element(
        page,
        "Column",
        region=RectangleRegion(0, 0, 0, 0, "img"),
        element_id="Column2",
        referenced_by=["Table1"],
        references=["Cell2"],
    )
    document.add_element(
        page,
        "Cell",
        region=RectangleRegion(0, 0, 0, 0, "img"),
        element_id="Cell1",
        referenced_by=["Row1", "Column1"],
    )

    type1 = SemanticType(uri="Type1", label="Type 1")
    type2 = SemanticType(uri="Type2", label="Type 2")
    entity1 = SemanticEntity(uri="Entity1", label="Entity 1", type=type1)
    entity2 = SemanticEntity(uri="Entity2", label="Entity 2", type=type2)
    document.semantic_network.add_entity(entity1)
    document.semantic_network.add_entity(entity2)
    document.semantic_network.add_relationship(
        head=entity1, tail=entity2, label="related_to"
    )

    document.semantic_references.add_reference("Column1", "Type1")
    document.semantic_references.add_reference("Column2", "Type2")
    document.semantic_references.add_reference("Cell1", "Entity1")
    document.semantic_references.add_reference("Cell2", "Entity2")

    return document


def test_draw(sample_graph):
    drawer = Drawer(sample_graph)
    try:
        drawer.draw(show_plot=False)  # Ensure no exceptions are raised
    except Exception as e:
        pytest.fail(f"Drawer.draw raised an exception: {e}")


def test_highlight_path_exists(sample_graph):
    drawer = Drawer(sample_graph)
    assert drawer.highlight_path("A", "C", show_plot=False) is True


def test_highlight_path_not_exists(sample_graph):
    drawer = Drawer(sample_graph)
    assert drawer.highlight_path("C", "D", show_plot=False) is False


def test_visualize_subgraph(sample_graph):
    drawer = Drawer(sample_graph)
    try:
        drawer.visualize_subgraph(
            ["A", "B", "C"], show_plot=False
        )  # Ensure no exceptions are raised
    except Exception as e:
        pytest.fail(f"Drawer.visualize_subgraph raised an exception: {e}")


def test_draw_document(sample_document):
    drawer = Drawer(sample_document)
    try:
        drawer.draw_separate_networks(
            show_plot=True,
            edge_labels=True,
            layout_func="planar",
        )  # Ensure no exceptions are raised
    except Exception as e:
        pytest.fail(f"Drawer.draw raised an exception for Document: {e}")


def test_draw_document_only_references(sample_document):
    drawer = Drawer(sample_document.references)
    try:
        drawer.draw(
            show_plot=True, edge_labels=True, layout_func="planar"
        )  # Ensure no exceptions are raised
    except Exception as e:
        pytest.fail(f"Drawer.draw raised an exception for Document: {e}")


def test_draw_document_only_semantic_references(sample_document):
    drawer = Drawer(sample_document.semantic_references)
    try:
        drawer.draw(
            show_plot=True, edge_labels=True, layout_func="planar"
        )  # Ensure no exceptions are raised
    except Exception as e:
        pytest.fail(f"Drawer.draw raised an exception for Document: {e}")


def test_draw_document_only_semantic_network(sample_document):
    drawer = Drawer(sample_document.semantic_network)
    try:
        drawer.draw(
            show_plot=True, edge_labels=True, layout_func="planar"
        )  # Ensure no exceptions are raised
    except Exception as e:
        pytest.fail(f"Drawer.draw raised an exception for Document: {e}")


def test_highlight_path_document(sample_document):
    drawer = Drawer(sample_document.references)
    assert drawer.highlight_path("page-0", "Column1") is True
    assert drawer.highlight_path("Table1", "Cell2") is True
    assert drawer.highlight_path("Column1", "Cell2") is False


def test_visualize_subgraph_document(sample_document):
    drawer = Drawer(sample_document.references)
    try:
        drawer.visualize_subgraph(["page-0", "Table1"])  # Ensure no exceptions are raised
    except Exception as e:
        pytest.fail(f"Drawer.visualize_subgraph raised an exception for Document: {e}")
