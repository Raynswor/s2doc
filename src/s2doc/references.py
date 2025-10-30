from collections import defaultdict

from .errors import LoadFromDictError


_EMPTY_FROZEN = frozenset()


class ReferenceGraph:
    def __init__(self):
        self.adj: dict[str, set[str]] = defaultdict(set)
        self.rev_adj: dict[str, set[str]] = defaultdict(set)
        self._roots: set[str] = set()
        self._non_roots: set[str] = set()

    @property
    def roots(self) -> set[str]:
        return self._roots

    def add_reference(self, parent: str, child: str) -> None:
        """
        Adds a reference between a parent and a child node in the adjacency structure.

        This method updates the adjacency dictionary (`self.adj`) to include the child
        node as a neighbor of the parent node. It also ensures that both the parent
        and child nodes exist in the adjacency dictionary, initializing them with
        empty sets if they are not already present.

        Args:
            parent (str): The parent node in the reference relationship.
            child (str): The child node in the reference relationship.

        Returns:
            None
        """
        if not isinstance(parent, str) or not isinstance(child, str):
            raise TypeError("Both parent and child must be strings.")

        if parent == child:
            return

        self.adj[parent].add(child)
        self.rev_adj[child].add(parent)

        self._non_roots.add(child)
        if parent not in self._non_roots:
            self._roots.add(parent)
        self._roots.discard(child)

        self._clean_transitive_edges(parent, child)

    def _clean_transitive_edges(self, parent: str, child: str) -> None:
        """
        Clean up redundant paths in the graph after adding a new edge.

        Removes any edge A->C if there already exists a path A->B->C.

        Args:
            parent (str): The parent node in the newly added reference.
            child (str): The child node in the newly added reference.
        """
        # Collect all edges to remove to avoid modifying sets during iteration
        edges_to_remove: list[tuple[str, str]] = []
        empty_nodes_to_clean: list[str] = []

        adj = self.adj
        rev = self.rev_adj

        # Check each grandparent (parents of parent)
        for grandparent in list(rev.get(parent, _EMPTY_FROZEN)):
            # If the grandparent already has a direct edge to child, it's redundant
            if child in adj.get(grandparent, _EMPTY_FROZEN):
                edges_to_remove.append((grandparent, child))

        # Check each grandchild (children of child)
        for grandchild in list(adj.get(child, _EMPTY_FROZEN)):
            # If the parent already has a direct edge to grandchild, it's redundant
            if grandchild != child and grandchild in adj.get(parent, _EMPTY_FROZEN):
                edges_to_remove.append((parent, grandchild))

        # Now safely remove all the collected edges
        for edge_parent, edge_child in edges_to_remove:
            parent_children = adj.get(edge_parent)
            if parent_children and edge_child in parent_children:
                parent_children.remove(edge_child)
                if not parent_children:
                    empty_nodes_to_clean.append(edge_parent)

            child_parents = rev.get(edge_child)
            if child_parents and edge_parent in child_parents:
                child_parents.remove(edge_parent)

        # Clean up empty entries from adj
        for node in empty_nodes_to_clean:
            if node in adj and not adj[node]:
                del adj[node]

    def contains(self, target: str) -> bool:
        """
        Determines whether the target string exists in the structure starting from any of the root nodes.

        This method performs a depth-first search (DFS) traversal starting from each root node
        to check if the target string is present. It keeps track of visited nodes to avoid
        infinite loops in case of cyclic references.

        Args:
            target (str): The string to search for in the structure.

        Returns:
            bool: True if the target string is found, False otherwise.
        """
        visited: set[str] = set()
        stack = list(self._roots)
        adj = self.adj
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in visited:
                continue
            visited.add(node)
            for child in adj.get(node, _EMPTY_FROZEN):
                if child not in visited:
                    stack.append(child)
        return False

    def _dfs(self, node: str, target: str, visited: set[str]) -> bool:
        # Defensive DFS that doesn't create new dict entries during reads
        if node == target:
            return True
        if node in visited:
            return False
        visited.add(node)
        for child in self.adj.get(node, _EMPTY_FROZEN):
            if self._dfs(child, target, visited):
                return True
        return False

    def get_children(self, parent: str) -> set[str]:
        """
        Retrieve the set of children nodes for a given parent node.

        Args:
            parent (str): The identifier of the parent node.

        Returns:
            set[str]: A set of identifiers representing the children nodes of the given parent.
                      If the parent node does not exist, an empty set is returned.
        """
        return set(self.adj.get(parent, _EMPTY_FROZEN))

    def get_nodes_with_non_zero_in_degree(self) -> set[str]:
        """
        Get all nodes with non-zero in-degree.

        Returns:
            set[str]: A set of all nodes with non-zero in-degree.
        """
        return set(self.rev_adj.keys())

    def get_parents(self, child: str) -> set[str]:
        """
        Retrieve the set of parent nodes for a given child node in a directed graph.

        Args:
            child (str): The child node for which to find parent nodes.

        Returns:
            set[str]: A set of parent nodes that have a directed edge to the given child node.
        """
        return set(self.rev_adj.get(child, _EMPTY_FROZEN))

    def remove_reference(self, parent: str, child: str) -> None:
        """
        Remove a reference between a parent and a child.

        Args:
            parent (str): The parent node.
            child (str): The child node.
        """
        parent_children = self.adj.get(parent)
        if parent_children and child in parent_children:
            parent_children.remove(child)

            # Check if this was the last parent before removing
            child_parents = self.rev_adj.get(child)
            was_last_parent = bool(
                child_parents and len(child_parents) == 1 and parent in child_parents
            )
            if child_parents:
                child_parents.remove(parent)

            # Clean up empty adjacency list
            if not parent_children:
                del self.adj[parent]

            # If child has no more parents, it becomes a root
            if was_last_parent:
                if child in self.rev_adj:
                    del self.rev_adj[child]
                self._roots.add(child)
                self._non_roots.discard(child)

    def remove_node(self, node: str) -> None:
        """
        Remove a node and all its references from the graph.

        Args:
            node (str): The node to remove.
        """
        # Collect nodes that will become roots to avoid modifying sets during iteration
        new_roots: list[str] = []
        adj = self.adj
        rev = self.rev_adj

        for child in list(adj.get(node, _EMPTY_FROZEN)):
            # Check if this node was the last parent before removing
            child_parents = rev.get(child)
            was_last_parent = bool(
                child_parents and len(child_parents) == 1 and node in child_parents
            )
            if child_parents:
                child_parents.remove(node)
            if was_last_parent:
                if child in rev:
                    del rev[child]
                new_roots.append(child)

        for parent in list(rev.get(node, _EMPTY_FROZEN)):
            parent_children = adj.get(parent)
            was_last_child = bool(
                parent_children
                and len(parent_children) == 1
                and node in parent_children
            )
            if parent_children:
                parent_children.remove(node)
            if was_last_child:
                if parent in adj:
                    del adj[parent]

        # Now safely update root tracking
        for new_root in new_roots:
            self._roots.add(new_root)
            self._non_roots.discard(new_root)

        adj.pop(node, None)
        rev.pop(node, None)
        self._roots.discard(node)
        self._non_roots.discard(node)

    def replace_node(self, old_node: str, new_node: str) -> None:
        """
        Replace a node with a new node in the graph.

        Args:
            old_node (str): The node to replace.
            new_node (str): The new node to insert.
        """
        adj = self.adj
        rev = self.rev_adj

        if old_node in adj:
            adj[new_node] = adj.pop(old_node)
            for child in adj[new_node]:
                parents = rev.get(child)
                if parents:
                    parents.discard(old_node)
                    parents.add(new_node)

        if old_node in rev:
            rev[new_node] = rev.pop(old_node)
            for parent in rev[new_node]:
                children = adj.get(parent)
                if children:
                    children.discard(old_node)
                    children.add(new_node)

        if old_node in self._roots:
            self._roots.remove(old_node)
            self._roots.add(new_node)
        if old_node in self._non_roots:
            self._non_roots.remove(old_node)
            self._non_roots.add(new_node)

    def _traverse(self, node: str, direction: str) -> set[str]:
        """
        Generic traversal method to get all related nodes in a specified direction.

        Args:
            node (str): The starting node for traversal.
            direction (str): The direction of traversal, either "descendants" or "ancestors".

        Returns:
            set[str]: A set of all related nodes in the specified direction.
        """
        related_nodes: set[str] = set()

        def _dfs(n: str) -> None:
            neighbors = (
                self.adj.get(n, _EMPTY_FROZEN)
                if direction == "descendants"
                else self.rev_adj.get(n, _EMPTY_FROZEN)
            )
            for neighbor in neighbors:
                if neighbor not in related_nodes:
                    related_nodes.add(neighbor)
                    _dfs(neighbor)

        _dfs(node)
        return related_nodes

    def get_descendants(self, node: str) -> set[str]:
        """
        Get all descendants of a given node.

        Args:
            node (str): The node to get descendants for.

        Returns:
            set[str]: A set of all descendant nodes.
        """
        return self._traverse(node, "descendants")

    def get_ancestors(self, node: str) -> set[str]:
        """
        Get all ancestors of a given node.

        Args:
            node (str): The node to get ancestors for.

        Returns:
            set[str]: A set of all ancestor nodes.
        """
        return self._traverse(node, "ancestors")

    def get_roots_of(self, node: str) -> set[str]:
        """
        Get all root nodes of a given node.

        Args:
            node (str): The node to get roots for.

        Returns:
            set[str]: A set of all root nodes.
        """
        roots: set[str] = self.get_ancestors(node)
        return roots & self.roots

    def get_all_nodes(self) -> set[str]:
        """
        Get all nodes in the graph.

        Returns:
            set[str]: A set of all nodes in the graph.
        """
        return set(self.adj.keys()) | set(self.rev_adj.keys())

    def to_obj(self) -> dict[str, list[str]]:
        """
        Export the graph as a dictionary.

        Returns:
            dict[str, list[str]]: A dictionary representation of the graph.
        """
        return {node: list(children) for node, children in self.adj.items()}

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> "ReferenceGraph":
        """
        Create a ReferenceGraph from a dictionary.

        Args:
            data (dict[str, list[str]]): A dictionary representation of the graph.

        Returns:
            ReferenceGraph: The constructed graph.
        """
        try:
            graph = cls()
            for parent, children in data.items():
                for child in children:
                    graph.add_reference(parent, child)
            return graph
        except Exception as e:
            raise LoadFromDictError(cls.__name__, str(e))
