import html
from collections.abc import Generator, Iterable
from typing import Any

from .element import Element
from .geometry import Region


class TableCell(Element):
    def __init__(
        self,
        oid: str,
        boundingBox: Region,
        data: dict[str, Any] | None = None,
        confidence: float | None = None,
    ):
        super().__init__(oid, "table_cell", boundingBox, data, confidence)

    def is_label(self) -> bool:
        """Returns True if the cell is marked as a header (row or column)."""
        return self.row_label or self.column_label
    def is_row_label(self) -> bool:
        """Returns True if the cell is marked as a row label."""
        return self.row_label
    def is_column_label(self) -> bool:
        """Returns True if the cell is marked as a column label."""
        return self.column_label
    def is_projected_row_label(self) -> bool:
        """Returns True if the cell is marked as a projected row label."""
        return self.data.get("projected_row_label", False)

    @property
    def row_label(self) -> bool:
        return self.data.get("row_label", False)

    @property
    def projected_row_label(self) -> bool:
        return self.data.get("pr_row_label", False)

    @property
    def column_label(self) -> bool:
        return self.data.get("column_label", False)

    @property
    def content(self) -> str:
        return self.data.get("content", "")

    @row_label.setter
    def row_label(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("row_label must be a boolean")
        self.data["row_label"] = value

    @column_label.setter
    def column_label(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("column_label must be a boolean")
        self.data["column_label"] = value

    @projected_row_label.setter
    def projected_row_label(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("projected_row_label must be a boolean")
        self.data["pr_row_label"] = value
        self.data["row_label"] = value  # projected row labels are also row labels

    @content.setter
    def content(self, value: str):
        if not isinstance(value, str):
            raise ValueError("content must be a string")
        self.data["content"] = value

    @classmethod
    def from_dict(cls, d: dict) -> "TableCell":
        return cls(
            d["oid"],
            Region.from_dict(d["r"]),
            data=d.get("data"),
            confidence=d.get("confidence"),
        )

    @classmethod
    def from_element(cls, element: Element) -> "TableCell":
        return cls(element.oid, element.region, element.data, element.confidence)

    def to_obj(self) -> dict:
        return super().to_obj()

class TableTuple:
    def __init__(
        self,
        row_header: list[str],
        column_header: list[str],
        value: str,
        origin: str,
        confidence: float,
        row_index: int = -1,
        column_index: int = -1,
    ):
        self.row_header = row_header
        self.column_header = column_header
        self.value = value
        self.origin = origin
        self.confidence = confidence
        self.row_index = row_index
        self.column_index = column_index

    @classmethod
    def from_dict(cls, d: dict) -> "TableTuple":
        if "t" in d:
            return cls(
                [cell for cell in d.get("t", [[], [], ""])[0]],
                [cell for cell in d.get("t", [[], [], ""])[1]],
                d.get("t", [[], [], ""])[2],
                d.get("o", ""),
                d.get("conf", 0.0),
            )
        elif "row" in d:
            return cls(
                d.get("row", []),
                d.get("column", []),
                d.get("value", ""),
                d.get("origin", ""),
                d.get("confidence", 0.0),
                d.get("row_index", -1),
                d.get("column_index", -1),
            )
        else:
            raise ValueError("Invalid TableTuple dict format")

    def to_obj(self) -> dict:
            return {
                "row": self.row_header,
                "column": self.column_header,
                "value": self.value,
                "origin": self.origin,
                "confidence": self.confidence,
                "row_index": self.row_index,
                "column_index": self.column_index,
            }



class Table(Element):
    def __init__(
        self,
        oid: str,
        boundingBox: Region,
        data: dict[str, Any] | None = None,
        confidence: float | None = None,
    ):
        super().__init__(oid, "table", boundingBox, data, confidence)
        #  - cell_nodes: a dict mapping a unique cell node id (e.g. "cell_0_0") to its properties,
        #       including the original cell value, a grid bounding box (top, left, bottom, right),
        #       and all grid positions (row, col) it covers.
        #     - coord_to_group: mapping of each (row, col) to the group index.
        #     - group_to_node: mapping of each group index to its unique cell node id.
        #     - n_rows, n_cols: dimensions of the 2D cell array.
        if data is not None:
            if "cells" in data:
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
            self.semantic_model: list[TableTuple] = [TableTuple.from_dict(t) if not isinstance( t, TableTuple) else t
                                   for t in self.data.get("semantic_model", [])]


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
        # convert cells from elements to ids, preserving span information
        cells_copy = []
        cells = self.cells
        for row_idx, row in enumerate(cells):
            row_copy: list = []
            for col_idx, cell in enumerate(row):
                if cell is None:
                    row_copy.append(None)
                elif isinstance(cell, str):
                    row_copy.append(cell)
                elif isinstance(cell, dict):
                    # Preserve rowSpan and colSpan if they exist in the dict
                    cell_id = cell.get("oid", "")
                    if "rowSpan" in cell or "colSpan" in cell:
                        # Store as dict with oid, rowSpan, colSpan
                        cell_entry = {"oid": cell_id}
                        if "rowSpan" in cell:
                            cell_entry["rowSpan"] = cell["rowSpan"]
                        if "colSpan" in cell:
                            cell_entry["colSpan"] = cell["colSpan"]
                        row_copy.append(cell_entry)
                    else:
                        row_copy.append(cell_id)
                else:
                    # Handle Element objects with rowSpan/colSpan attributes
                    cell_id = cell.oid
                    if hasattr(cell, 'rowSpan') or hasattr(cell, 'colSpan'):
                        cell_entry = {"oid": cell_id}
                        if hasattr(cell, 'rowSpan') and cell.rowSpan and cell.rowSpan > 1:
                            cell_entry["rowSpan"] = cell.rowSpan
                        if hasattr(cell, 'colSpan') and cell.colSpan and cell.colSpan > 1:
                            cell_entry["colSpan"] = cell.colSpan
                        row_copy.append(cell_entry if cell_entry != {"oid": cell_id} else cell_id)
                    else:
                        row_copy.append(cell_id)
            cells_copy.append(row_copy)
        di = super().to_obj()
        di["data"]["cells"] = cells_copy
        di["data"]["semantic_model"] = [t.to_obj() if isinstance(t, TableTuple) else t for t in self.semantic_model]
        # di["cells"] = cells_copy
        return di

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
        n_rows = self.n_rows
        n_cols = self.n_cols
        cells = self.cells
        for c in range(n_cols):
            yield [cells[r][c] for r in range(n_rows)]

    ## Semantic Model
    @property
    def semantic_model(self) -> Iterable[TableTuple]:
        return self.data.get("semantic", [])

    @semantic_model.setter
    def semantic_model(self, value: Iterable[TableTuple] | Iterable[dict]):
        if all(isinstance(v, TableTuple) for v in value):
            self.data["semantic"] = [v.to_obj() for v in value]  # type: ignore
        elif all(isinstance(v, dict) for v in value):
            self.data["semantic"] = list(value)  # type: ignore
        else:
            raise ValueError("semantic_model must be Iterable[TableTuple] or Iterable[dict]")

    ## Logical Model (Grid)
    @property
    def cells(self) -> list[list[str]]:
        return self.data.get("cells", [])

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

        visited: set[tuple[int, int]] = set()
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
                if isinstance(cell_value, str):
                    node_id = cell_value
                elif isinstance(cell_value, (TableCell, Element)):
                    node_id = cell_value.oid
                elif isinstance(cell_value, dict):
                    node_id = cell_value.get("oid", str(cell_value))
                else:
                    # Fallback for any other type
                    node_id = str(cell_value)

                groups.append((node_id, group_coords))

        # Build unique cell nodes for each group.
        cell_nodes: dict[str, dict] = {}
        group_to_node: dict[int, str] = {}

        # [TODO] Bug?

        for i, (node_id, coords) in enumerate(groups):

            node_id = node_id if isinstance(node_id, str) else node_id.oid

            top = min(r for r, _ in coords)
            left = min(c for _, c in coords)
            bottom = max(r for r, _ in coords)
            right = max(c for _, c in coords)
            cell_nodes[node_id] = {
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

            # Map each coordinate to this group index
            for coord in coords:
                coord_to_group[coord] = i

        return cell_nodes, coord_to_group, group_to_node, n_rows, n_cols

    def get_cell_at(self, row: int, col: int) -> tuple[str, dict]:
        """
        Returns the unique cell (as a dict) that covers the given grid position.
        If the cell spans multiple positions, the same cell info is returned.
        """
        if not (0 <= row < self.n_rows and 0 <= col < self.n_cols):
            raise IndexError("Cell position out of range")
        group_index = self.coord_to_group.get((row, col))
        if group_index is None:
            return "", {}
        node_id = self.group_to_node[group_index]
        return node_id, self.cell_nodes[node_id]

    def get_rows_cols_of_cell(self, cell_id: str) -> list[tuple[int, int]]:
        """
        Returns all grid positions (row, col) covered by the given cell id.
        """
        positions: list[tuple[int, int]] = []
        # iterate once and check both common representations (string id or dict with 'oid')
        for r in range(self.n_rows):
            row = self.cells[r]
            for c in range(self.n_cols):
                val = row[c]
                if val == cell_id:
                    positions.append((r, c))
                elif isinstance(val, dict) and val.get("oid") == cell_id:
                    positions.append((r, c))
        return positions

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
                cell_id, cell_info = self.get_cell_at(row_idx, col_idx)

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

                    # Add cell content (handle different value types). Use html.escape
                    # to properly escape special characters.
                    try:
                        cell_text = "".join(doc.get_element_data_value(cell_id))
                    except Exception as _:
                        cell_text = ""
                    cell_content = html.escape(str(cell_text))
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
                raw_val, cell_info = self.get_cell_at(row_idx, col_idx)

                if not cell_info or not raw_val or raw_val == "" or raw_val.startswith("empty"):
                    # Empty cell
                    row_data.append("")
                    continue

                # Add cell content (handle different value types)
                try:
                    cell_text = "".join(doc.get_element_data_value(raw_val))
                except Exception as _:
                    cell_text = ""
                if cell_text is None:
                    cell_text = ""
                # Escape pipe and normalize whitespace for markdown
                cell_content = (
                    str(cell_text)
                    .replace("|", "\\|")
                    .replace("\n", " ")
                    .replace("\r", " ")
                    .strip()
                )

                # For merged cells, use content in the first occurrence and placeholders for others
                grid = cell_info["grid"]
                if row_idx == grid["top"] and col_idx == grid["left"]:
                    row_data.append(cell_content)
                    for r in range(grid["top"], grid["bottom"] + 1):
                        for c in range(grid["left"], grid["right"] + 1):
                            if r != row_idx or c != col_idx:
                                rendered_positions.add((r, c))
                elif (row_idx, col_idx) in rendered_positions:
                    row_data.append("^")
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

    def build_semantic_model(self, doc) -> list[TableTuple]:
        if not self.cells:
            self.semantic_model = []
            self.data["semantic_model"] = []
            return []

        H = len(self.cells)
        W = len(self.cells[0]) if H > 0 else 0

        def push_run(lst, cid):
            if not lst or lst[-1] != cid:
                lst.append(cid)

        def dedup_concat(*lists):
            out, seen = [], set()
            for lst in lists:
                for cid in lst or ():
                    if cid not in seen:
                        seen.add(cid)
                        out.append(cid)
            return out

        col_headers = [[] for _ in range(W)]
        col_meta = [[] for _ in range(W)]
        col_meta_seen = [set() for _ in range(W)]
        last_row_hdr = [None for _ in range(W)]

        tuples: list[TableTuple] = []

        for i in range(H):
            left = []
            row_meta = []
            row_meta_seen = set()

            for j in range(W):
                try:
                    c: Element = self.cells[i][j]
                except IndexError:
                    continue
                if c is None or isinstance(c, str):
                    continue

                if c["data"].get('row_label'):
                    push_run(left, c["oid"])
                    for k in col_headers[j]:
                        if k not in row_meta_seen:
                            row_meta_seen.add(k)
                            row_meta.append(k)
                    last_row_hdr[j] = c["oid"]

                if c["data"].get('column_label'):
                    push_run(col_headers[j], c["oid"])
                    for k in left:
                        if k not in col_meta_seen[j]:
                            col_meta_seen[j].add(k)
                            col_meta[j].append(k)

                if not c["data"].get('row_label') and not c["data"].get('column_label'):
                    proj = [last_row_hdr[j]] if last_row_hdr[j] else []
                    rh = dedup_concat(left, proj, row_meta)
                    ch = dedup_concat(col_headers[j], col_meta[j])
                    tuples.append(TableTuple(rh, ch, c["oid"], origin=self.oid, confidence=float(self.confidence or 0.0)))

        self.semantic_model = tuples
        self.data["semantic_model"] = self.semantic_model
        return tuples



def get_table_obj(document, t: Table | Element | str) -> Table:
    if isinstance(t, str):
        table: Table = Table.from_element(document.get_element_obj(t))
    elif isinstance(t, Element):
        table = Table.from_element(t)
    else:
        table = t
    return table


def generate_table_cell_matrix(document, t: Table | Element | str) -> list[list[TableCell | None]]:
    table: Table = get_table_obj(document, t)

    obj_matrix: list[list[(TableCell | None)]] = [[None for _ in range(table.n_cols)] for _ in range(table.n_rows)]

    for node_id, cell in table.cell_nodes.items():
        obj = document.get_element_obj(node_id)
        if obj is None:
            continue
        if not isinstance(obj, TableCell):
            obj = TableCell.from_element(obj)
        grid = cell.get("grid", {})
        for r in range(grid.get("top", 0), grid.get("bottom", 0) + 1):
            for c in range(grid.get("left", 0), grid.get("right", 0) + 1):
                obj_matrix[r][c] = obj
    return obj_matrix
