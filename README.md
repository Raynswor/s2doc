# S2Doc

**S2Doc** (Spatial-Semantic Document) is data format (and Python library) designed for representing documents in general document processing and analysis pipelines. It provides a unified structure that captures both spatial layout and semantic content and is designed to be easily extensible and adjustable.

S2Doc originated as an internal data format developed for a table extraction pipeline, where existing document representations proved insufficient for capturing both the spatial layout and semantic structure required for robust processing. Recognizing the lack of a standard format for this class of tasks, S2Doc was generalized to serve as a unified, extensible framework for spatial-semantic document representation, aiming to fill this gap.

## Features

- **Standardized Document Format**: A consistent data model for document representation.
- **Page Management**: Create and manage pages with image and coordinate space metadata.
- **Element Handling**: Define and manipulate content elements with bounding boxes and categories.
- **Semantic Layer**: Link document elements to semantic types and entities.
- **Reference Graphs**: Represent structural, spatial, and semantic relationships between document elements.
- **Serialization**: Convert documents to and from JSON for storage, transfer, or interoperability.
- **Extensibility**: Designed to support custom extensions for new annotation types, layout features, and processing components.

## Installation

To install the library, run:

```bash
git clone
cd s2doc
pip install s2doc
```

## Usage

### Creating a Document

```python
from s2doc.document import Document
from s2doc.page import Page

doc = Document(oid="doc-1")
page = Page(oid="page-1", number=1, None, {
    "img": Space(label="img", dimensions=[100.0, 100.0], axis_directions=[True, False]),
})
doc.add_page(page)
```

### Adding Elements

```python
from s2doc.geometry import RectangleRegion

reg = RectangleRegion(x1=0, y1=0, x2=100, y2=100, space="img")
element_id = doc.add_element(page="page-1", category="Text", region=reg)
```


## Contributing

Contributions are welcome. To contribute, fork the repository and submit a pull request with a clear description of your changes.


## Citation
TBD

