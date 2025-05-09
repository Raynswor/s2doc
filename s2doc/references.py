from collections import defaultdict

from .errors import LoadFromDictError


class ReferenceGraph:
    def __init__(self):
        self.adj: dict[str, set[str]] = defaultdict(set)
        self._roots: set[str] = set()
        self.non_roots: set[str] = set()

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
        self.adj[parent].add(child)
        self.non_roots.add(child)
        if parent not in self.non_roots:
            self._roots.add(parent)
        self._roots.discard(child)

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
        for root in self._roots:
            if self._dfs(root, target, visited):
                return True
        return False

    def _dfs(self, node: str, target: str, visited: set[str]) -> bool:
        if node == target:
            return True
        if node in visited:
            return False
        visited.add(node)
        return any(self._dfs(child, target, visited) for child in self.adj[node])

    def get_children(self, parent: str) -> set[str]:
        """
        Retrieve the set of children nodes for a given parent node.

        Args:
            parent (str): The identifier of the parent node.

        Returns:
            set[str]: A set of identifiers representing the children nodes of the given parent.
                      If the parent node does not exist, an empty set is returned.
        """
        return self.adj.get(parent, set())

    def get_nodes_with_non_zero_in_degree(self) -> set[str]:
        """
        Get all nodes with non-zero in-degree.

        Returns:
            set[str]: A set of all nodes with non-zero in-degree.
        """
        return {child for children in self.adj.values() for child in children}

    def get_parents(self, child: str) -> set[str]:
        """
        Retrieve the set of parent nodes for a given child node in a directed graph.

        Args:
            child (str): The child node for which to find parent nodes.

        Returns:
            set[str]: A set of parent nodes that have a directed edge to the given child node.
        """
        return {parent for parent, children in self.adj.items() if child in children}

    def remove_reference(self, parent: str, child: str) -> None:
        """
        Remove a reference between a parent and a child.

        Args:
            parent (str): The parent node.
            child (str): The child node.
        """
        if parent in self.adj and child in self.adj[parent]:
            self.adj[parent].remove(child)
            if not self.adj[parent]:
                del self.adj[parent]
            if not any(child in children for children in self.adj.values()):
                self._roots.add(child)

    def remove_node(self, node: str) -> None:
        """
        Remove a node and all its references from the graph.

        Args:
            node (str): The node to remove.
        """
        self.adj.pop(node, None)
        self._roots.discard(node)
        for children in self.adj.values():
            children.discard(node)

    def replace_node(self, old_node: str, new_node: str) -> None:
        """
        Replace a node with a new node in the graph.

        Args:
            old_node (str): The node to replace.
            new_node (str): The new node to insert.
        """
        if old_node in self.adj:
            self.adj[new_node] = self.adj.pop(old_node)
        for parent in list(self.adj.keys()):
            if old_node in self.adj[parent]:
                self.adj[parent].remove(old_node)
                self.adj[parent].add(new_node)
        if old_node in self._roots:
            self._roots.remove(old_node)
            self._roots.add(new_node)
        if old_node in self.non_roots:
            self.non_roots.remove(old_node)
            self.non_roots.add(new_node)

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
                self.adj.get(n, set())
                if direction == "descendants"
                else self.get_parents(n)
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
        return set(self.adj.keys()) | {c for cs in self.adj.values() for c in cs}

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
