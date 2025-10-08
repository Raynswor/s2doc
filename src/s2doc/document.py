import datetime
import json
import logging
import uuid
from collections.abc import Callable, Generator, Iterable
from typing import Any, overload
from typing import Literal as L

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image
from shapely import box

from .base import DocObj
from .element import Element
from .errors import (
    AreaNotFoundError,
    DocumentError,
    ExistenceError,
    LoadFromDictError,
    PageNotFoundError,
)
from .font import Font
from .geometry import Region
from .normalizedObject import NormalizedObj
from .page import Page
from .references import ReferenceGraph
from .revision import Revision
from .semantics import Literal, SemanticEntity, SemanticKnowledgeGraph, SemanticType
from .space import Space
from .util import base64_to_img, img_to_base64

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)


class Document(DocObj):
    def __init__(
        self,
        oid: str,
        pages: NormalizedObj[Page] | None = None,
        elements: NormalizedObj[Element] | None = None,
        references: ReferenceGraph | None = None,
        revisions: list[Revision] | None = None,
        fonts: list[Font] | None = None,
        semantic_network: SemanticKnowledgeGraph | None = None,
        semantic_references: ReferenceGraph | None = None,
        metadata: dict[str, Any] | None = None,
        raw_data: bytes | None = None,
    ):
        self.oid: str = oid
        self.pages: NormalizedObj[Page] = pages or NormalizedObj[Page]()
        self.elements: NormalizedObj[Element] = elements or NormalizedObj[Element]()
        self.fonts: list[Font] = fonts or []

        # manages connections between spatial elements (elements)
        # --> always is_part_of/has_subpart or something like that
        self.references: ReferenceGraph = references or ReferenceGraph()

        # Create initial revision if none provided
        self.revisions: list[Revision] = revisions or [
            Revision(
                datetime.datetime.now().isoformat(timespec="milliseconds"),
                set(),
                comment="Initial Revision",
            )
        ]

        # manages available Types (classes), Entities (instances) and Relations
        self.semantic_network: SemanticKnowledgeGraph = (
            semantic_network or SemanticKnowledgeGraph()
        )

        # manages the connections between spatial elements (elements) and semantic entities
        # always is_of_class OR is_individual relations
        self.semantic_references: ReferenceGraph = (
            semantic_references or ReferenceGraph()
        )

        self.metadata: dict[str, Any] = metadata or {}
        self.raw_data: bytes | None = raw_data

        # Cache the ID generation function
        self.id_generation_variant: Callable[..., str] = self._generate_element_id(
            "long"
        )

    def _generate_element_id(self, variant: str) -> Callable[..., str]:
        """Generate a function that creates element IDs based on the specified variant."""
        # Create a UUID generator with a small cache for better performance
        uuid_cache = []

        def get_uuid():
            if not uuid_cache:
                # Pre-generate a small batch of UUIDs for better performance
                uuid_cache.extend(str(uuid.uuid4())[:8] for _ in range(5))
            return uuid_cache.pop()

        def generate_long_id(page, category):
            return (
                f"{self.oid}-{self.pages[page].number}-{category.lower()}-{get_uuid()}"
            )

        variants = {
            "uuid": lambda page, category: get_uuid(),
            "document": lambda page, category: f"{self.oid}-{get_uuid()}",
            "page": lambda page, category: f"{self.pages[page].number}-{get_uuid()}",
            "category": lambda page, category: f"{category.lower()}-{get_uuid()}",
            "long": generate_long_id,
        }

        if variant not in variants:
            raise ValueError(f"Unknown id generation variant: {variant}")

        return variants[variant]

    def add_element(
        self,
        page: str | Page,
        category: str,
        region: Region,
        element_id: str | None = None,
        data: Any = None,
        referenced_by: str | list[str] | None = None,
        references: list[str] | None = None,
        confidence: float | None = None,
        convert_to_xml: bool = False,
    ) -> str:
        """
        Adds a element to the document.

        Args:
            page (str | Page): The page identifier or Page object where the element will be added.
            category (str): The category of the element.
            region (Region): The bounding box defining the element's location and size.
            element_id (str | None, optional): The unique identifier for the element. If not provided,
                one will be generated. Defaults to None.
            data (Any, optional): Additional data associated with the element. Defaults to None.
            referenced_by (str | list[str] | None, optional): Identifier(s) of the object(s) referencing
                this element. Defaults to None.
            references (list[str] | None, optional): List of identifiers that this element references.
                Defaults to None.
            confidence (float | None, optional): Confidence score for the element. Defaults to None.
            convert_to_xml (bool, optional): Whether to convert the bounding box to XML space.
                Defaults to False.

        Returns:
            str: The unique identifier of the added element.

        Raises:
            PageNotFoundError: If the specified page does not exist.
            DocumentError: If the category is not provided or the bounding box is not an instance
                of Region.
        """
        if not category:
            raise DocumentError("Category must be provided.")
        if not isinstance(region, Region):
            raise DocumentError("region must be an instance of Region.")

        if isinstance(page, str):
            page_obj: Page | None = self.pages.get(key=page)
            if not page_obj:
                raise PageNotFoundError(f"Page '{page}' does not exist.")
        elif isinstance(page, Page):
            page_obj: Page = page

        if convert_to_xml:
            region = region.convert_space(
                page_obj.factor_between_spaces("img", "xml"), "xml"
            )
        space: Space | None = page_obj.spaces.get(region.space)
        if not space:
            raise DocumentError(
                f"Space '{region.space}' does not exist in page '{page_obj.oid}'."
            )
        # see if region is within the space dimensions TODO: currently only 2D space and RectangleRegion
        if not (
            0 <= region.bounds[0] <= space.dimensions[0]
            and 0 <= region.bounds[2] <= space.dimensions[0]
            and 0 <= region.bounds[1] <= space.dimensions[1]
            and 0 <= region.bounds[3] <= space.dimensions[1]
        ):
            logging.warning(
                f"Region {region} is out of bounds for space '{region.space}' in page '{page_obj.oid}'."
            )
            return ""
            # raise DocumentError(
            #     f"Region '{region}' is out of bounds for space '{region.space}' in page '{page_obj.oid}'."
            # )

        category = category.lower()

        while not element_id or element_id in self.elements:
            element_id = self.id_generation_variant(page_obj.oid, category)

        ar = Element.create(
            element_id,
            region=region,
            category=category,
            data=data or {},
            confidence=confidence,
        )
        self.elements.add(ar)
        # self.references.add_reference(page, element_id)
        self.revisions[-1].objects.add(element_id)

        if referenced_by:
            if isinstance(referenced_by, str):
                self.references.add_reference(referenced_by, element_id)
            else:
                for r in referenced_by:
                    self.references.add_reference(r, element_id)
        else:
            self.references.add_reference(page_obj.oid, element_id)

        if references:
            if isinstance(references, str):
                self.references.add_reference(element_id, references)
            else:
                for r in references:
                    self.references.add_reference(element_id, r)

        return element_id

    def delete_elements(self, element_ids: list[str]):
        """
        Deletes the specified elements and their associated references from the document.

        Args:
            element_ids (list[str]): A list of element IDs to be deleted.

        Side Effects:
            - Removes the specified elements from the `self.elements` collection
            - Removes the corresponding nodes from references and semantic_references
            - Updates the `del_objs` set in the latest revision with the deleted element IDs
        """
        # Use NormalizedObj's batch removal for better performance
        if hasattr(self.elements, "remove_multiple"):
            self.elements.remove_multiple(element_ids)
        else:
            for element_id in element_ids:
                self.elements.remove(element_id)

        # Remove references for each element
        for element_id in element_ids:
            self.references.remove_node(element_id)
            self.semantic_references.remove_node(element_id)

        # Batch update the revision's deletion record
        self.revisions[-1].del_objs.update(element_ids)

    def delete_element(self, element_id: str):
        """
        Deletes a element and its associated references from the document.

        Args:
            element_id (str): The identifier of the element to be deleted.

        Side Effects:
            - Removes the element from the `elements` list.
            - Removes the corresponding node from `references` and `semantic_references`.
            - Adds the element ID to the `del_objs` set in the latest revision.

        Raises:
            KeyError: If the element_id does not exist in the `elements` list or references.
        """
        self.elements.remove(element_id)
        self.references.remove_node(element_id)
        self.semantic_references.remove_node(element_id)

        self.revisions[-1].del_objs.add(element_id)

    def replace_element(self, old_id: str, new_element: Element) -> None:
        assert self.elements.byId[old_id].category == new_element.category.lower()
        self.elements.byId[new_element.oid] = new_element
        self.elements.remove(old_id)

        self.references.replace_node(old_id, new_element.oid)

        self.revisions[-1].del_objs.add(old_id)

    def _element_id_to_object(self, element_id_or_obj: str | Element) -> Element:
        if isinstance(element_id_or_obj, str):
            element = self.elements.get(element_id_or_obj)
            if not element:
                raise AreaNotFoundError(f"Element '{element_id_or_obj}' not found")
            return element
        return element_id_or_obj

    def _obj_id_to_str(self, element_id_or_obj: str | Element | Page) -> str:
        if not isinstance(element_id_or_obj, str) and hasattr(element_id_or_obj, "oid"):
            return element_id_or_obj.oid
        return element_id_or_obj

    def replace_element_multiple(
        self, old_id: str, new_elements: list[Element]
    ) -> None:
        page = self.find_page_of_element(old_id, False)
        self.replace_element(old_id, new_elements[0])
        for a in new_elements[1:]:
            self.add_element(
                page, a.category, a.region, data=a.data, confidence=a.confidence
            )

    @overload
    def find_page_of_element(
        self, element: str | Element, as_string: L[True]
    ) -> str: ...

    @overload
    def find_page_of_element(
        self, element: str | Element, as_string: L[False]
    ) -> Page: ...

    def find_page_of_element(
        self, element: str | Element, as_string: bool = True
    ) -> str | Page:
        """
        If an element can be found on multiple pages, only the first one found is returned. it is not necessarily the first one in the document.
        """
        element_id = self._obj_id_to_str(element)
        roots = list(self.references.get_roots_of(element_id))
        for r in roots:
            if r in self.pages.allIds:
                if as_string:
                    return r
                else:
                    return self.pages[r]
        raise ValueError(f"Element '{element_id}' not found in any page references.")

    def get_latest_elements(self) -> list[str]:
        """Get the list of element IDs from the latest revision.

        Returns:
            list[str]: List of current element IDs after applying all revisions.
        """
        # Use a more descriptive variable name and pre-allocate for efficiency
        element_set = set()
        # Process revisions in order
        for rev in self.revisions:
            element_set.update(rev.adjust_objs(element_set))
        return list(element_set)

    def get_visible_elements(
        self, element_id: str | Element, direction: str, category: str | None = None
    ) -> list[Element]:
        """
        Retrieves a list of elements visible from a given element in a specified direction
        and belonging to a specific category.

        Args:
            element_id (str | Element): The identifier or object of the element to start from.
            direction (str): The direction to look for visible elements.
                            Valid values are "left", "right", "above", and "below".
            category (str): The category of elements to filter the results.
                            If None, category of source is considered.

        Returns:
            list[Element]: A list of elements visible from the specified element in the given
                        direction and matching the specified category. Returns an empty
                        list if no elements are found or if the direction is invalid.

        Raises:
            ValueError: If the specified element is not found on any page.
        """
        source = self._element_id_to_object(element_id)
        page = self.find_page_of_element(source, False)
        if not page:
            raise ValueError(f"Element '{element_id}' not found on any page.")

        bbox = source.region
        space = page.spaces[bbox.space]
        if not len(space.dimensions) == 2:
            raise DocumentError(
                f"Invalid space dimensions for get_visible_elements: {space.dimensions}"
            )

        axis_directions = space.axis_directions

        source_shape = source.region._shape
        candidates = self.get_element_type(category or source.category, page.oid)
        # Precompute page-local element ids to avoid scanning the whole document
        page_elem_ids = set(self.references.get_descendants(page.oid))

        def is_visible(target):
            target_shape = target.region._shape
            if direction == "left":
                if axis_directions[0]:  # x-axis goes left to right
                    if source_shape.bounds[0] <= target_shape.bounds[2]:
                        return False
                else:  # x-axis goes right to left
                    if source_shape.bounds[0] >= target_shape.bounds[2]:
                        return False
                corridor = box(
                    target_shape.bounds[2],
                    max(source_shape.bounds[1], target_shape.bounds[1]),
                    source_shape.bounds[0],
                    min(source_shape.bounds[3], target_shape.bounds[3]),
                )
            elif direction == "right":
                if axis_directions[0]:  # x-axis goes left to right
                    if source_shape.bounds[2] >= target_shape.bounds[0]:
                        return False
                else:  # x-axis goes right to left
                    if source_shape.bounds[2] <= target_shape.bounds[0]:
                        return False
                corridor = box(
                    source_shape.bounds[2],
                    max(source_shape.bounds[1], target_shape.bounds[1]),
                    target_shape.bounds[0],
                    min(source_shape.bounds[3], target_shape.bounds[3]),
                )
            elif direction == "above":
                if axis_directions[1]:  # y-axis goes top to bottom
                    if source_shape.bounds[1] >= target_shape.bounds[3]:
                        return False
                else:  # y-axis goes bottom to top
                    if source_shape.bounds[1] <= target_shape.bounds[3]:
                        return False
                corridor = box(
                    max(source_shape.bounds[0], target_shape.bounds[0]),
                    target_shape.bounds[3],
                    min(source_shape.bounds[2], target_shape.bounds[2]),
                    source_shape.bounds[1],
                )
            elif direction == "below":
                if axis_directions[1]:  # y-axis goes top to bottom
                    if source_shape.bounds[3] <= target_shape.bounds[1]:
                        return False
                else:  # y-axis goes bottom to top
                    if source_shape.bounds[3] >= target_shape.bounds[1]:
                        return False
                corridor = box(
                    max(source_shape.bounds[0], target_shape.bounds[0]),
                    source_shape.bounds[3],
                    min(source_shape.bounds[2], target_shape.bounds[2]),
                    target_shape.bounds[1],
                )
            else:
                return False

            if not corridor.intersects(target_shape):
                return False
            # Only check elements on the same page to determine corridor obstruction
            for eid in page_elem_ids:
                # fetch element once
                el = self.elements.get(eid)
                if not el:
                    continue
                # skip source/target
                if (
                    el is source
                    or el is target
                    or el.oid == source.oid
                    or el.oid == target.oid
                ):
                    continue
                if corridor.intersects(el.region._shape):
                    return False
            return True

        return [el for el in candidates if is_visible(el) and el != source]

        # x_shift, y_shift = 0, 0

        # if direction == "left" and bbox.x1 > 0:
        #     x_shift = -1
        # elif direction == "right" and bbox.x2 < space.width:
        #     x_shift = 1
        # elif direction == "above" and bbox.y1 > 0:
        #     y_shift = -1
        # elif direction == "below" and bbox.y2 < space.height:
        #     y_shift = 1
        # else:
        #     return []

        # return self.get_elements_at_position(
        #     page, bbox.x1 + x_shift, bbox.y1 + y_shift, category
        # )

    def get_elements_at_position(
        self, page: str, x: int | float, y: int | float, category: str
    ) -> list[Element]:
        if page not in self.pages.allIds:
            raise ExistenceError(f"Page '{page}' does not exist.")
        return [
            element
            for element_id in self.references.get_descendants(page)
            if (element := self.elements[element_id]).category == category.lower()
            and element.region.x1 <= x <= element.region.x2
            and element.region.y1 <= y <= element.region.y2
        ]

    def get_element_obj(self, element_id: str) -> Element:
        return self.elements[element_id]

    def get_element_data_value(
        self, element: str | Element, val: str = "content"
    ) -> list[Any]:
        """Get a element's data value or fall back to descendants."""
        element = self._element_id_to_object(element)
        if not element:
            raise AreaNotFoundError(f"Element '{element}' not found")

        g = element.data.get(val, None)
        if not g:
            refs = self.references.get_children(element.oid)
            if refs:
                # Sort children by their spatial position (top-to-bottom, left-to-right)
                child_elements = []
                for ref in refs:
                    child_element = self.elements.get(ref)
                    if child_element:
                        child_elements.append((ref, child_element))

                child_elements.sort(
                    key=lambda x: (x[1].region.bounds[1], x[1].region.bounds[0])
                )

                return [
                    " ".join(t)
                    for ref, _ in child_elements
                    if (t := self.get_element_data_value(ref, val)) is not None
                ]
        return [g] if g else []

    def add_revision(self, name: str):
        """Add a revision with the given name."""
        self.revisions.append(
            Revision(
                datetime.datetime.now().isoformat(timespec="milliseconds"),
                set(),
                comment=name,
            )
        )

    def add_page(
        self, page: Page | None = None, img: str | np.ndarray | None = None
    ) -> str:
        if page:
            self.pages.add(page)
            return page.oid

        if img:
            open_img = Image.open(img) if isinstance(img, str) else Image.fromarray(img)
            page = Page(
                f"page-{len(self.pages.allIds)}",
                len(self.pages.allIds),
                img_to_base64(open_img),
                spaces={
                    "img": Space(
                        "img", [open_img.width, open_img.height], [True, False]
                    )
                },
            )
        else:
            page = Page(
                f"page-{len(self.pages.allIds)}",
                len(self.pages.allIds),
                None,
                spaces={"none": Space("none", [0, 0], [True, False])},
            )

        self.pages.add(page)
        return page.oid

    def get_element_type(
        self, category: str | list[str], page: str | int | Page = ""
    ) -> Iterable[Element]:
        """
        Get all elements of a specific category, optionally filtered by page.

        Args:
            category: The element category to filter by
            page: Optional page to limit the search scope

        Returns:
            Iterable[Element]: Generator of matching elements
        """
        if isinstance(category, list):
            category = [cat.lower() for cat in category]
            return self.get_element_by(lambda x: x.category in category, page)
        return self.get_element_by(lambda x: x.category == category.lower(), page)

    def get_element_by(
        self, fun: Callable[[Element], bool], page: str | int | Page = ""
    ) -> Iterable[Element]:
        """
        Filter elements using a provided function, optionally within a page scope.

        Args:
            fun: Function that takes an Element and returns a boolean
            page: Optional page identifier to limit the search scope

        Returns:
            Iterable[Element]: Generator of elements that match the filter
        """
        # Normalize page reference to a string id
        page_id = ""
        if isinstance(page, Page):
            page_id = page.oid
        elif page in self.pages.allIds:
            page_id = page
        elif isinstance(page, int) and f"page-{page}" in self.pages.allIds:
            page_id = f"page-{page}"

        # Three possible search paths with early returns for efficiency
        if page and not page_id:
            return []  # Invalid page reference
        elif not page:
            # Search all elements (no page filter)
            return (elem for elem in self.elements.values() if fun(elem))
        else:
            # Search only elements within a specific page
            descendants = self.references.get_descendants(page_id)
            return (
                self.elements[elem_id]
                for elem_id in descendants
                if fun(self.elements[elem_id])
            )

    def get_img_snippets(
        self,
        elements: list[Element],
        padding: tuple[int, int] = (0, 0),
        page: Page | str | None = None,
    ) -> Generator[ArrayLike, None, None]:
        if not page:
            p = self.find_page_of_element(elements[0], False)
        elif isinstance(page, Page):
            p = page
        elif isinstance(page, str):
            p = self.pages.get(page)
            if not p:
                raise PageNotFoundError(f"Page '{page}' does not exist.")

        if p.img is None:
            raise DocumentError(f"Page {p} does not have an image")
        img = np.array(base64_to_img(p.img))  # type: ignore

        for element in elements:
            bb = element.region.convert_space(
                p.factor_between_spaces("xml", "img"), "img"
            )
            yield img[
                int(bb.y1) - padding[1] : int(bb.y2) + padding[1],
                int(bb.x1) - padding[0] : int(bb.x2) + padding[0],
                :,
            ]

    @overload
    def get_img_snippet(
        self,
        element_id: str | Element,
        as_string: L[False],
        padding: tuple[int, int] = (0, 0),
    ) -> Image.Image: ...

    @overload
    def get_img_snippet(
        self,
        element_id: str | Element,
        as_string: L[True],
        padding: tuple[int, int] = (0, 0),
    ) -> str: ...

    def get_img_snippet(
        self,
        element_id: str | Element,
        as_string: bool = True,
        padding: tuple[int, int] = (0, 0),
    ) -> str | Image.Image:
        element = self._element_id_to_object(element_id)
        if not element:
            raise AreaNotFoundError(f"Element {element_id} does not exist")
        p = self.find_page_of_element(element, False)
        return self.get_img_snippet_from_bb(element.region, p, as_string, padding)

    def get_img_snippet_from_bb(
        self,
        bb: Region,
        p: str | Page,
        as_string: bool,
        padding: tuple[int, int] = (0, 0),
    ) -> str | Image.Image:
        if not bb:
            raise DocumentError(f"Invalid bounding box {bb}")

        if not isinstance(p, Page):
            p: Page = self.pages.get(p)
        else:
            p: Page = p

        bb = bb.convert_space(p.factor_between_spaces("xml", "img"), "img").rectify()

        img: Image.Image = base64_to_img(p.img) if isinstance(p.img, str) else p.img
        cropped = Image.fromarray(
            np.array(img)[
                max(int(bb.y1) - padding[1], 0) : min(
                    int(bb.y2) + padding[1], img.size[1] - 1
                ),
                max(int(bb.x1) - padding[0], 0) : min(
                    int(bb.x2) + padding[0], img.size[0] - 1
                ),
                :,
            ]
        )
        return cropped if not as_string else img_to_base64(cropped)

    @overload
    def get_img_page(self, page: str, as_base64_string: L[True]) -> str: ...

    @overload
    def get_img_page(self, page: str, as_base64_string: L[False]) -> Image.Image: ...

    def get_img_page(
        self, page: str, as_base64_string: bool = True
    ) -> str | Image.Image:
        p = page if page in self.pages.allIds else ""
        return (
            (
                self.pages[p].img
                if as_base64_string
                else base64_to_img(self.pages[p].img)
            )
            if p
            else ""
        )

    def transpose_page(self, page_id: str):
        for obj in self.references.get_descendants(page_id):
            self.elements[obj].region.transpose()

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        """Create Document instance from a dictionary."""
        if (
            not isinstance(d, dict)
            or not d.get("oid")
            or "pages" not in d
            or "elements" not in d
        ):
            raise LoadFromDictError(cls.__name__, f"Invalid input: {d}")

        try:
            return cls(
                oid=d["oid"],
                pages=NormalizedObj.from_dict(d["pages"], "pages"),
                elements=NormalizedObj.from_dict(d["elements"], "elements"),
                references=(
                    ReferenceGraph.from_dict(d["references"])
                    if d.get("references")
                    else ReferenceGraph()
                ),
                revisions=[Revision.from_dict(x) for x in d.get("revisions", set())],
                fonts=[Font.from_dict(x) for x in d.get("fonts", [])]
                if d.get("fonts")
                else [],
                metadata=d.get("metadata", {}),
                raw_data=d.get("raw_data"),
                semantic_network=SemanticKnowledgeGraph.from_dict(
                    d["semantic_network"]
                ),
                semantic_references=(
                    ReferenceGraph.from_dict(d["semantic_references"])
                    if d.get("semantic_references")
                    else ReferenceGraph()
                ),
            )
        except Exception as e:
            raise LoadFromDictError(cls.__name__, str(e))

    def to_obj(self) -> dict:
        """Convert Document to a dictionary."""
        return {
            "oid": self.oid,
            "pages": self.pages.to_obj(),
            "elements": self.elements.to_obj(),
            "references": self.references.to_obj(),
            "revisions": [x.to_obj() for x in self.revisions],
            "fonts": [x.to_obj() for x in self.fonts],
            "metadata": self.metadata,
            "raw_data": self.raw_data,
            "semantic_network": self.semantic_network.to_obj(),
            "semantic_references": self.semantic_references.to_obj(),
        }

    def to_json(self) -> str:
        """
        Serialize Document to JSON.

        Returns:
            str: JSON representation of the document
        """

        # Custom encoder to properly handle numpy types, dates, and other special objects
        class NpEncoder(json.JSONEncoder):
            def default(self, o):
                # Cache imports for better performance
                import base64

                # Handle numpy types
                if isinstance(o, np.integer):
                    return int(o)
                if isinstance(o, np.floating):
                    return float(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()

                # Handle other special types
                if isinstance(o, datetime.datetime):
                    return o.isoformat()
                if isinstance(o, set):
                    return list(o)
                if isinstance(o, bytes):
                    try:
                        return o.decode("utf-8")
                    except UnicodeDecodeError:
                        return base64.b64encode(o).decode("utf-8")
                if isinstance(o, Element):
                    return o.to_obj()

                # Fallback to default encoder
                return super(NpEncoder, self).default(o)

        # Convert to object dictionary then serialize to JSON
        return json.dumps(self.to_obj(), cls=NpEncoder)

    def is_referenced_by(self, element: Element | str, element2: Element | str) -> bool:
        """
        Determines if a element is referenced by another element.

        Args:
            element (Element | str): The element or its identifier to check for references.
            element2 (Element | str): The element or its identifier to check as a potential ancestor.

        Returns:
            bool: True if `element2` is an ancestor of `element`, False otherwise.
        """
        # Convert to string IDs for consistency
        element_id = self._obj_id_to_str(element)
        ancestor_id = self._obj_id_to_str(element2)

        # Get ancestors and check for presence - early return if no ancestors
        ancestors = self.references.get_ancestors(element_id)
        if not ancestors:
            return False

        # Direct check for presence in ancestors
        return ancestor_id in ancestors

    """
    Semantics
    """

    def _assert_element_exists(self, element_id: str) -> None:
        if element_id not in self.elements:
            raise KeyError(f"Element '{element_id}' not found")

    def _assert_entity_exists(self, uri: str) -> None:
        if uri not in self.semantic_network.entities:
            raise KeyError(f"Entity '{uri}' not loaded in semantic network")

    def _assert_type_exists(self, uri: str) -> None:
        if uri not in self.semantic_network.available_types:
            raise KeyError(f"Type '{uri}' not available in semantic network")

    def _annotate(self, element_id: str, reference: str) -> None:
        """Core reference‐addition after validation."""
        self._assert_element_exists(element_id)
        self.semantic_references.add_reference(element_id, reference)

    def annotate_element_with_uri(self, element_id: str, uri: str) -> None:
        self._annotate(element_id, uri)

    def annotate_element_with_entity(
        self, element_id: str, entity: SemanticEntity
    ) -> None:
        self._assert_entity_exists(entity.uri)
        self._annotate(element_id, entity.uri)

    def annotate_element_with_type(self, element_id: str, typ: SemanticType) -> None:
        self._assert_type_exists(typ.uri)
        self._annotate(element_id, typ.uri)

    def annotate_element_with_literal(
        self, element_id: str, literal: Literal | str
    ) -> None:
        """
        Annotate an element with a literal value.

        The ReferenceGraph stores only strings, so Literal instances are
        converted into their lexical form "value^^datatype".

        Args:
            element_id: The ID of the element to annotate
            literal: The literal value (string or Literal object)

        Raises:
            KeyError: If the element doesn't exist
            TypeError: If literal is not of the expected type
        """
        # Validate element existence
        self._assert_element_exists(element_id)

        # Convert string to default-typed Literal if needed
        if isinstance(literal, str):
            literal = Literal(literal)
        elif not isinstance(literal, Literal):
            raise TypeError(f"Expected Literal or str, got {type(literal).__name__}")

        # Convert to lexical form (e.g., '"foo"^^xsd:string' or '42^^xsd:integer')
        lexical = str(literal)

        # Add the reference
        self.semantic_references.add_reference(element_id, lexical)

    # def get_elements_by_entity(self, entity: SemanticEntity) -> list[Element]:
    #     return [self.elements[aid] for aid, ents in self.semantic_references.items() if entity in ents]

    # def get_elements_by_type(self, semantic_type_uri: str) -> list[Element]:
    #     return [
    #         self.elements[aid]
    #         for aid, ents in self.semantic_references.byId.items()
    #         for ent in ents
    #         if isinstance(ent.type, SemanticType) and ent.type.uri == semantic_type_uri
    #     ]
