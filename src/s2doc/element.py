from collections.abc import Generator, Set
from typing import Any

from .base import DocObj
from .errors import IncompatibleError, LoadFromDictError
from .geometry import Region


class Element(DocObj):
    def __init__(
        self,
        oid: str,
        category: str,
        region: Region,
        data: dict[str, Any] | None = None,
        confidence: float | None = None,
    ):
        self.oid: str = oid
        self.category: str = category
        self.region: Region = region
        self.data: dict[str, Any] = data if data else {}
        self.confidence: float | None = confidence

    @classmethod
    def create(
        cls,
        oid: str,
        category: str,
        region: Region,
        data: dict[str, Any] | None = None,
        confidence: float | None = None,
    ) -> "Element":
        """Factory method to instantiate the correct subclass."""
        if category.lower() == "table":
            return Table(oid, region, data, confidence)
        return cls(oid, category, region, data, confidence)

    def __repr__(self) -> str:
        return f"<{self.oid}: {self.region} {self.data if self.data else ''}>"

    def merge(
        self, other: "Element", merge_data: bool = True, merge_confidence: bool = True
    ):
        # check if the other element is of the same type
        if self.category != other.category:
            raise IncompatibleError("type", self.category, other.category)

        self.region = self.region.union(other.region)
        if merge_data:
            for k, v in other.data.items():
                if k in self.data:
                    if isinstance(self.data[k], list):
                        self.data[k].extend(v)
                    elif isinstance(self.data[k], dict):
                        self.data[k].update(v)
                    elif isinstance(self.data[k], str):
                        self.data[k] += v
                    else:
                        self.data[k] = v
                else:
                    self.data[k] = v
        if merge_confidence:
            if self.confidence is not None and other.confidence is not None:
                self.confidence = max(self.confidence, other.confidence)
            elif other.confidence is not None:
                self.confidence = other.confidence

    @classmethod
    def from_dict(cls, d: dict) -> "Element":
        try:
            return cls.create(
                d["oid"],
                d["c"],
                Region.from_dict(d["r"]),
                data=d.get("data"),
                confidence=d.get("confidence"),
            )
        except KeyError as e:
            raise LoadFromDictError(cls.__name__, str(e))

    def to_obj(self) -> dict:
        dic: dict[str, Any] = {
            "oid": self.oid,
            "c": self.category,
            "r": self.region.to_obj(),
        }
        if self.data:
            dic["data"] = self.data
        if self.confidence is not None:
            dic["confidence"] = self.confidence
        return dic


class Table(Element):
    def __init__(
        self,
        oid: str,
        boundingBox: Region,
        data: dict[str, Any] | None = None,
        confidence: float | None = None,
    ):
        # Ensure that data contains the "cells" key.
        super().__init__(oid, "table", boundingBox, data, confidence)
        if data is not None and "cells" in data:
            (
                self.cell_nodes,
                self.coord_to_group,
                self.group_to_node,
                self.n_rows,
                self.n_cols,
            ) = self._compute_cell_groups()
        else:
            self.data["cells"] = []
            self.cell_nodes = {}
            self.coord_to_group = {}
            self.group_to_node = {}
            self.n_rows = 0
            self.n_cols = 0

    @staticmethod
    def from_element(element: Element):
        return Table(element.oid, element.region, element.data, element.confidence)

    @classmethod
    def from_dict(cls, d: dict) -> "Table":
        return cls(
            d["oid"],
            Region.from_dict(d["r"]),
            data=d.get("data"),
            confidence=d.get("confidence"),
        )

    def to_obj(self):
        # convert cells from elements to ids
        cells_copy = []
        for i in range(len(self.cells)):
            cells_copy.append([])
            for j in range(len(self.cells[i])):
                cell = self.cells[i][j]
                if cell is None:
                    cells_copy[i].append(None)
                else:
                    cells_copy[i].append(
                        cell.oid if isinstance(cell, Element) else cell
                    )
        di = super().to_obj()
        di["cells"] = cells_copy
        return di

    @property
    def cells(self) -> list[list[str]]:
        return self.data.get("cells", [])

    @property
    def caption(self) -> str:
        return self.data.get("caption", "")

    @property
    def number(self) -> str:
        return self.data.get("number", "")

    @property
    def rows(self) -> Generator[list[str], None, None]:
        for row in self.cells:
            yield row

    @property
    def columns(self) -> Generator[list[str], None, None]:
        for c in range(self.n_cols):
            column = []
            for r in range(self.n_rows):
                column.append(self.cells[r][c])
            yield column

    @cells.setter
    def cells(self, value: list[list[str]]):
        self.data["cells"] = value
        (
            self.cell_nodes,
            self.coord_to_group,
            self.group_to_node,
            self.n_rows,
            self.n_cols,
        ) = self._compute_cell_groups()

    def _compute_cell_groups(
        self,
    ) -> tuple[dict[str, dict], dict[tuple[int, int], int], dict[int, str], int, int]:
        """
        Groups contiguous grid positions (using 4-connected neighbors) that
        share the same cell id. Each group represents a unique cell.

        Returns:
            - cell_nodes: a dict mapping a unique cell node id (e.g. "cell_0_0") to its properties,
              including the original cell value, a grid bounding box (top, left, bottom, right),
              and all grid positions (row, col) it covers.
            - coord_to_group: mapping of each (row, col) to the group index.
            - group_to_node: mapping of each group index to its unique cell node id.
            - n_rows, n_cols: dimensions of the 2D cell array.
        """
        cells_array = self.data["cells"]
        n_rows = len(cells_array)
        n_cols = len(cells_array[0]) if n_rows > 0 else 0

        visited: Set[tuple[int, int]] = set()
        groups: list[
            tuple[Any, list[tuple[int, int]]]
        ] = []  # Each group: (cell_value, list of positions)
        coord_to_group: dict[tuple[int, int], int] = {}

        # Walk over every position and group contiguous positions of the same id.
        for r in range(n_rows):
            for c in range(n_cols):
                if (r, c) in visited:
                    continue
                cell_value = cells_array[r][c]
                if not cell_value:
                    continue
                stack = [(r, c)]
                group_coords = []
                while stack:
                    rr, cc = stack.pop()
                    if (rr, cc) in visited:
                        continue
                    if cells_array[rr][cc] == cell_value:
                        visited.add((rr, cc))
                        group_coords.append((rr, cc))
                        # Check 4-connected neighbors.
                        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            rrr, ccc = rr + dr, cc + dc
                            if 0 <= rrr < n_rows and 0 <= ccc < n_cols:
                                if (rrr, ccc) not in visited and cells_array[rrr][
                                    ccc
                                ] == cell_value:
                                    stack.append((rrr, ccc))
                groups.append((cell_value, group_coords))
                for pos in group_coords:
                    coord_to_group[pos] = len(groups) - 1

        # Build unique cell nodes for each group.
        cell_nodes: dict[str, dict] = {}
        group_to_node: dict[int, str] = {}
        for i, (cell_value, coords) in enumerate(groups):
            top = min(r for r, _ in coords)
            left = min(c for _, c in coords)
            bottom = max(r for r, _ in coords)
            right = max(c for _, c in coords)
            node_id = f"cell_{top}_{left}"  # Use the top-left coordinate to identify the cell.
            cell_nodes[node_id] = {
                "node_id": node_id,
                "value": cell_value,
                # Here, the grid bounding box is defined by row/column indices.
                "grid": {
                    "top": top,
                    "left": left,
                    "bottom": bottom,
                    "right": right,
                },
                "positions": coords,
            }
            group_to_node[i] = node_id

        return cell_nodes, coord_to_group, group_to_node, n_rows, n_cols

    def get_cell_at(self, row: int, col: int) -> dict:
        """
        Returns the unique cell (as a dict) that covers the given grid position.
        If the cell spans multiple positions, the same cell info is returned.
        """
        if not (0 <= row < self.n_rows and 0 <= col < self.n_cols):
            raise IndexError("Cell position out of range")
        group_index = self.coord_to_group.get((row, col))
        if group_index is None:
            return {}
        node_id = self.group_to_node[group_index]
        return self.cell_nodes[node_id]

    def cells_to_graph(self) -> dict[str, Any]:
        """
        Converts the 2D cell array into a structure graph.

        The graph is represented as a dictionary with:
          - "nodes": mapping node ids to node properties.
          - "edges": a list of edges (each a dict with keys "from", "to", and "relation").

        The graph includes:
          - A table node.
          - One node per unique cell.
          - "hasCell" edges from the table node to each cell.
          - "nextInRow"/"prevInRow" edges between adjacent cells in each row.
          - "nextInColumn"/"prevInColumn" edges between adjacent cells in each column.
        """
        graph: dict[str, Any] = {"nodes": {}, "edges": []}
        # Create the table node.
        table_node_id = self.oid if self.oid else "table"
        graph["nodes"][table_node_id] = {
            "type": "table",
            "id": table_node_id,
            "rows": self.n_rows,
            "columns": self.n_cols,
        }
        # Add cell nodes and link them to the table.
        for node_id, node in self.cell_nodes.items():
            graph["nodes"][node_id] = {"type": "cell", **node}
            graph["edges"].append(
                {
                    "from": table_node_id,
                    "to": node_id,
                    "relation": "hasCell",
                }
            )

        # Build row-based connections.
        # For each row, collect the cells that cover that row.
        rows_nodes: dict[int, list[dict]] = {r: [] for r in range(self.n_rows)}
        for node in self.cell_nodes.values():
            bb = node["grid"]
            for r in range(bb["top"], bb["bottom"] + 1):
                rows_nodes[r].append(node)
        # For each row, connect adjacent cells.
        for r, nodes in rows_nodes.items():
            # Remove duplicates (a spanning cell may appear in more than one row).
            unique_nodes = {node["node_id"]: node for node in nodes}.values()
            ordered = sorted(unique_nodes, key=lambda n: n["grid"]["left"])
            for i in range(len(ordered) - 1):
                from_node = ordered[i]["node_id"]
                to_node = ordered[i + 1]["node_id"]
                graph["edges"].append(
                    {
                        "from": from_node,
                        "to": to_node,
                        "relation": "nextInRow",
                    }
                )
                graph["edges"].append(
                    {
                        "from": to_node,
                        "to": from_node,
                        "relation": "prevInRow",
                    }
                )

        # Build column-based connections.
        cols_nodes: dict[int, list[dict]] = {c: [] for c in range(self.n_cols)}
        for node in self.cell_nodes.values():
            bb = node["grid"]
            for c in range(bb["left"], bb["right"] + 1):
                cols_nodes[c].append(node)
        for c, nodes in cols_nodes.items():
            unique_nodes = {node["node_id"]: node for node in nodes}.values()
            ordered = sorted(unique_nodes, key=lambda n: n["grid"]["top"])
            for i in range(len(ordered) - 1):
                from_node = ordered[i]["node_id"]
                to_node = ordered[i + 1]["node_id"]
                graph["edges"].append(
                    {
                        "from": from_node,
                        "to": to_node,
                        "relation": "nextInColumn",
                    }
                )
                graph["edges"].append(
                    {
                        "from": to_node,
                        "to": from_node,
                        "relation": "prevInColumn",
                    }
                )

        return graph

    def to_html(self, doc) -> str:
        """
        Converts the table's cell matrix to an HTML table structure.

        Uses the computed cell groups to properly handle merged cells with
        rowspan and colspan attributes.

        Returns:
            str: HTML table string with proper structure
        """
        if not self.cells or self.n_rows == 0 or self.n_cols == 0:
            return "<table></table>"

        html_parts = ["<table>"]

        # Track which cells have already been rendered (for merged cells)
        rendered_positions: set[tuple[int, int]] = set()

        for row_idx in range(self.n_rows):
            html_parts.append("  <tr>")

            for col_idx in range(self.n_cols):
                # Skip if this position is part of a merged cell already rendered
                if (row_idx, col_idx) in rendered_positions:
                    continue

                # Get the cell at this position
                cell_info = self.get_cell_at(row_idx, col_idx)

                if not cell_info:
                    # Empty cell
                    html_parts.append("    <td></td>")
                    rendered_positions.add((row_idx, col_idx))
                else:
                    # Get cell dimensions for rowspan/colspan
                    grid = cell_info["grid"]
                    rowspan = grid["bottom"] - grid["top"] + 1
                    colspan = grid["right"] - grid["left"] + 1

                    # Build the cell tag with appropriate attributes
                    cell_tag = "    <td"
                    if rowspan > 1:
                        cell_tag += f' rowspan="{rowspan}"'
                    if colspan > 1:
                        cell_tag += f' colspan="{colspan}"'
                    cell_tag += ">"

                    # Add cell content (handle different value types)
                    cell_value = "".join(
                        doc.get_element_data_value(cell_info.get("value", ""))
                    )
                    if isinstance(cell_value, str):
                        cell_content = cell_value
                    else:
                        cell_content = str(cell_value) if cell_value is not None else ""

                    # Escape HTML special characters in cell content
                    cell_content = (
                        cell_content.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace('"', "&quot;")
                        .replace("'", "&#x27;")
                    )

                    html_parts.append(f"{cell_tag}{cell_content}</td>")

                    # Mark all positions covered by this merged cell as rendered
                    for r in range(grid["top"], grid["bottom"] + 1):
                        for c in range(grid["left"], grid["right"] + 1):
                            rendered_positions.add((r, c))

            html_parts.append("  </tr>")

        html_parts.append("</table>")
        return "\n".join(html_parts)

    def to_markdown(self, doc) -> str:
        """
        Converts the table's cell matrix to a Markdown table structure.

        Uses the computed cell groups to properly handle merged cells.
        Note: Markdown doesn't support rowspan/colspan, so merged cells
        will be represented by repeating content or using placeholders.

        Returns:
            str: Markdown table string
        """
        if not self.cells or self.n_rows == 0 or self.n_cols == 0:
            return ""

        markdown_parts = []

        # Track which cells have already been rendered (for merged cells)
        rendered_positions: set[tuple[int, int]] = set()

        # Store the actual table data for markdown generation
        table_data: list[list[str]] = []

        for row_idx in range(self.n_rows):
            row_data = []

            for col_idx in range(self.n_cols):
                # Get the cell at this position
                cell_info = self.get_cell_at(row_idx, col_idx)

                if not cell_info:
                    # Empty cell
                    row_data.append("")
                else:
                    # Add cell content (handle different value types)
                    cell_value = "".join(
                        doc.get_element_data_value(cell_info.get("value", ""))
                    )
                    if isinstance(cell_value, str):
                        cell_content = cell_value
                    else:
                        cell_content = str(cell_value) if cell_value is not None else ""

                    # Escape markdown special characters in cell content
                    cell_content = (
                        cell_content.replace("|", "\\|")
                        .replace("\n", " ")
                        .replace("\r", " ")
                        .strip()
                    )

                    # For merged cells, we'll use the content in the first occurrence
                    # and empty strings for subsequent positions
                    grid = cell_info["grid"]
                    if row_idx == grid["top"] and col_idx == grid["left"]:
                        # This is the top-left position of a merged cell
                        row_data.append(cell_content)

                        # Mark all positions covered by this merged cell as rendered
                        for r in range(grid["top"], grid["bottom"] + 1):
                            for c in range(grid["left"], grid["right"] + 1):
                                if (
                                    r != row_idx or c != col_idx
                                ):  # Skip the current position
                                    rendered_positions.add((r, c))
                    elif (row_idx, col_idx) in rendered_positions:
                        # This position is part of a merged cell, use placeholder
                        row_data.append("^")  # Indicates merged cell continuation
                    else:
                        row_data.append(cell_content)

            table_data.append(row_data)

        # Generate markdown table
        if not table_data:
            return ""

        # Create the header row (first row of data)
        header = "| " + " | ".join(table_data[0]) + " |"
        markdown_parts.append(header)

        # Create the separator row
        separator = "| " + " | ".join(["---"] * self.n_cols) + " |"
        markdown_parts.append(separator)

        # Add the remaining data rows
        for row_data in table_data[1:]:
            row = "| " + " | ".join(row_data) + " |"
            markdown_parts.append(row)

        return "\n".join(markdown_parts)
