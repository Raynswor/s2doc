# S2Doc

**S2Doc** (Spatial-Semantic Document) is data format (and Python library) designed for representing documents in general document processing and analysis pipelines. It provides a unified structure that captures both spatial layout and semantic content and is designed to be easily extensible and adjustable.

S2Doc originated as an internal data format developed for a table extraction pipeline, where existing document representations proved insufficient for capturing both the spatial layout and semantic structure required for robust processing. Recognizing the lack of a standard format for this class of tasks, S2Doc was generalized to serve as a unified, extensible framework for spatial-semantic document representation, aiming to fill this gap.

For more details, see the [arXiv paper](https://arxiv.org/abs/2511.01113).

[KIETA](https://github.com/Raynswor/kieta) is a framework built on top of S2Doc for document understanding pipelines with a focus on tables. It provides ready-to-use components for converting PDFs and images into the S2Doc format, serveral processing modules for table detection and structure recognition, and general analysis tasks as well as several exporters.

## Features

- Standardized document model (pages, elements, regions, semantics)
- Support for multiple coordinate spaces per page (xml/img/tokens)
- Region types: rectangles, polygons, lines, polylines, spans
- Serialization to/from compact JSON-like lists/dicts
- Reference graphs for spatial/semantic relations
- Extensible: add new element types, semantic entities, and converters

## Quickstart (developer)

Clone and install in editable mode (recommended for development):

```bash
git clone <repo-url>
cd s2doc
uv sync
```

This installs the package and development extras (pytest, ruff, pre-commit).

Run tests:

```bash
uv run pytest -q
```

Run the linters / auto-fixers:

```bash
uv run ruff check .
# or (auto-fix)
uv run ruff format .

# Run pre-commit hooks (applies fixes where configured)
uv run pre-commit run --all-files
```

API examples
------------

Create a document and add a rectangular element:

```python
from s2doc.document import Document
from s2doc.page import Page
from s2doc.geometry import RectangleRegion

doc = Document(oid="doc-1")
page = Page(oid="page-1", number=1, img=None)
page.spaces["img"].dimensions = [800.0, 600.0]
doc.pages.add(page)

reg = RectangleRegion(0, 0, 100, 50, space="img")
element_id = doc.add_element(page=page, category="Text", region=reg)
```

See the `tests/` directory for more usage examples that mirror common workflows.

Contributing
------------

Contributions are welcome. Please:

1. Fork the repository and create a feature branch
2. Run tests and linters locally (see Quickstart)
3. Open a pull request with a short description and tests for behavior changes


Reference
------------------
If you use S2Doc in your research, please cite the following paper:

```
@misc{kempf2025s2docspatialsemanticdocument,
      title={S2Doc -- Spatial-Semantic Document Format}, 
      author={Sebastian Kempf and Frank Puppe},
      year={2025},
      eprint={2511.01113},
      archivePrefix={arXiv},
      primaryClass={cs.DL},
      url={https://arxiv.org/abs/2511.01113}, 
}
```
