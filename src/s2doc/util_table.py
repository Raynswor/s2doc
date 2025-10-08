from . import Document, Element, Table, TableCell, TableTuple


def get_table_obj(document: Document, t: Table | Element | str) -> Table:
    if isinstance(t, str):
        table: Table = Table.from_element(document.get_element_obj(t))
    elif isinstance(t, Element):
        table = Table.from_element(t)
    else:
        table = t
    return table


def generate_table_cell_matrix(document: Document, t: Table | Element | str) -> list[list[TableCell | None]]:
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


def generate_semantic_model(document: Document, t: Table | Element | str) -> list[TableTuple]:
    table: Table = get_table_obj(document, t)

    tuples: list[TableTuple] = []

    for node_id, cell in table.cell_nodes.items():
        obj = document.get_element_obj(node_id)
        if obj is None:
            continue
        if not isinstance(obj, TableCell):
            obj = TableCell.from_element(obj)
        if obj.is_label():
            continue

        grid = cell.get("grid", {})

        # get the row and column header cells
        row_headers: list[TableCell] = []
        col_headers: list[TableCell] = []

        # [TODO] projected row header

        for r in range(grid.get("top", 0), grid.get("bottom", 0) + 1):
            for c in range(grid.get("left", 0), grid.get("right", 0) + 1):
                # find row header
                for cc in range(0, table.n_cols):
                    candidate = document.get_element_obj(table.get_cell_at(r, cc)[0])
                    if candidate is None:
                        continue
                    if not isinstance(candidate, TableCell):
                        candidate = TableCell.from_element(candidate)
                    if candidate.is_row_label():
                        if candidate not in row_headers:
                            row_headers.append(candidate)
                        break
                # find column header
                for rr in range(0, table.n_rows):
                    candidate = document.get_element_obj(table.get_cell_at(rr, c)[0])
                    if candidate is None:
                        continue
                    if not isinstance(candidate, TableCell):
                        candidate = TableCell.from_element(candidate)
                    if candidate.is_column_label():
                        if candidate not in col_headers:
                            col_headers.append(candidate)
                        break
        tuples.append(TableTuple(row_headers, col_headers, obj, origin=table.oid, confidence=0))
    return tuples