import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .document import Document
from .references import ReferenceGraph
from .semantics import SemanticEntity, SemanticKnowledgeGraph


class Drawer:
    """
    A general class for visualizing ReferenceGraph or SemanticNetwork using NetworkX and Matplotlib.
    """

    DEFAULT_COLORS = {
        "document": {
            "node": "lightblue",
            "edge": "gray",
            "root": "lightgreen",
            "entity": "lightblue",
            "type": "orange",
        },
        "semantic": {
            "node": "lightblue",
            "edge": "blue",
            "type": "orange",
        },
        "highlight": {
            "node": "yellow",
            "edge": "red",
        },
        "cross": {
            "edge": "red",
        },
    }

    def __init__(self, drawing_obj: ReferenceGraph | SemanticKnowledgeGraph | Document):
        """
        Initialize the Drawer with a graph.

        Args:
            graph (ReferenceGraph | SemanticNetwork | Document): The graph to visualize.
        """
        self.drawing_obj = drawing_obj
        self._cached_layouts = {}
        self._cached_colors = {}

        # Convert to NetworkX graph(s)
        self.nx_graph, self.semantic_graph = self._convert_to_networkx(self.drawing_obj)

    def _convert_to_networkx(
        self, drawing_obj
    ) -> tuple[nx.DiGraph, nx.DiGraph | None]:
        """
        Convert the graph to a NetworkX directed graph.

        Returns:
            nx.DiGraph or tuple[nx.DiGraph, nx.DiGraph]: A NetworkX directed graph
            representing the input graph, or a tuple of document and semantic graphs.
        """
        if isinstance(drawing_obj, ReferenceGraph):
            G = nx.DiGraph()
            # Add all nodes, including those with no connections
            all_nodes = drawing_obj.get_all_nodes()
            G.add_nodes_from(all_nodes)

            # Add all edges in one step with attributes
            G.add_edges_from(
                [
                    (parent, child, {"label": "contains"})
                    for parent, children in drawing_obj.adj.items()
                    for child in children
                ]
            )
            return G, None

        elif isinstance(drawing_obj, SemanticKnowledgeGraph):
            G = nx.DiGraph()

            # Add entities and types as nodes in batches
            nodes_to_add = []
            edges_to_add = []

            # Process all entities in one pass
            for ent in drawing_obj.entities.values():
                nodes_to_add.append((ent.uri, {"label": ent.label}))

                # Handle entity type relation
                if ent.type is not None:
                    if isinstance(ent.type, str):
                        type_id = ent.type
                        type_label = ent.type
                    else:
                        type_id = ent.type.uri
                        type_label = ent.type.label

                    nodes_to_add.append((type_id, {"label": type_label}))
                    edges_to_add.append((ent.uri, type_id, {"label": "is_a"}))

            # Add all relationships
            for rel in drawing_obj.relationships:
                head = (
                    rel.head.uri if isinstance(rel.head, SemanticEntity) else rel.head
                )
                tail = (
                    rel.tail.uri if isinstance(rel.tail, SemanticEntity) else rel.tail
                )
                edges_to_add.append((head, tail, {"label": rel.label}))

            # Batch add nodes and edges
            G.add_nodes_from(nodes_to_add)
            G.add_edges_from(edges_to_add)

            return G, None

        elif isinstance(drawing_obj, Document):
            # Create separate graphs for document and semantic network
            G_ref, _ = self._convert_to_networkx(drawing_obj.references)
            G_sem, _ = self._convert_to_networkx(drawing_obj.semantic_network)

            # Return both graphs for more flexibility in visualization
            return G_ref, G_sem

        # Default fallback
        return nx.DiGraph(), None

    def get_layout(self, graph, layout_name="spring", **kwargs):
        """Get a cached layout or compute a new one.

        Args:
            graph (nx.Graph): Graph to lay out
            layout_name (str): Name of layout algorithm
            **kwargs: Additional arguments for layout function

        Returns:
            dict: Node positions dictionary
        """
        # Create a unique key for this graph+layout combination
        graph_id = id(graph)
        cache_key = (graph_id, layout_name, frozenset(kwargs.items()))

        # Check cache first
        if cache_key in self._cached_layouts:
            return self._cached_layouts[cache_key]

        # Calculate new layout
        layout_func = self._get_layout_function(layout_name)
        positions = layout_func(graph, **kwargs)

        # Cache and return
        self._cached_layouts[cache_key] = positions
        return positions

    def _get_layout_function(self, layout_name: str = "spring"):
        """Get the appropriate NetworkX layout function.

        Args:
            layout_name (str): Name of the layout algorithm to use.

        Returns:
            callable: NetworkX layout function
        """
        layout_funcs = {
            "spring": nx.spring_layout,
            "circular": nx.circular_layout,
            "random": nx.random_layout,
            "shell": nx.shell_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "planar": nx.planar_layout,
        }
        return layout_funcs.get(layout_name.lower(), nx.spring_layout)

    def _finalize_plot(self, title=None, save_path=None, show_plot=True):
        """Common finishing steps for all plot types.

        Args:
            title (str, optional): Title for the plot
            save_path (str, optional): Path to save the figure
            show_plot (bool): Whether to show the plot
        """
        if title:
            plt.title(title, fontsize=16)

        plt.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _add_legend(
        self, legend_dict, loc="upper right", fontsize=8, ncol=1, bbox_to_anchor=None
    ):
        """Add a legend to the current plot.

        Args:
            legend_dict (dict): Dictionary mapping colors to labels
            loc (str): Legend location
            fontsize (int): Font size for legend
            ncol (int): Number of columns in legend
            bbox_to_anchor (tuple, optional): Anchor point for legend
        """
        if not legend_dict:
            return

        handles = [
            plt.Line2D( # type: ignore
                [0],
                [0],
                marker="o",
                color="w",
                label=label,
                markersize=10,
                markerfacecolor=color,
            )
            for color, label in legend_dict.items()
        ]

        legend_args = {"handles": handles, "loc": loc, "fontsize": fontsize}

        if bbox_to_anchor:
            legend_args["bbox_to_anchor"] = bbox_to_anchor

        if ncol > 1:
            legend_args["ncol"] = ncol

        plt.legend(**legend_args)

    def _draw_graph_elements(
        self,
        graph,
        pos,
        node_size,
        node_color,
        edge_color,
        font_size,
        edge_labels=False,
        edge_label_color="black",
    ):
        """
        Helper method to draw graph elements including nodes, edges, and labels.

        Args:
            graph (nx.Graph): The graph to draw.
            pos (dict): Node positions.
            node_size (int): Size of the nodes.
            node_color (list or str): Color(s) of the nodes.
            edge_color (str): Color of the edges.
            font_size (int): Font size for node labels.
            edge_labels (bool): Whether to show edge labels.
            edge_label_color (str): Color of the edge labels.
        """
        nx.draw_networkx_nodes(
            graph, pos, node_size=node_size, node_color=node_color, alpha=0.8
        )
        nx.draw_networkx_edges(graph, pos, edge_color=edge_color, width=1.5, alpha=0.7)
        nx.draw_networkx_labels(
            graph,
            pos,
            font_size=font_size,
            font_weight="bold",
            bbox={"ec": "k", "fc": "white", "alpha": 0.4},
            font_color="black",
        )
        if edge_labels:
            edge_label_dict = nx.get_edge_attributes(graph, "label")
            if edge_label_dict:
                nx.draw_networkx_edge_labels(
                    graph, pos, edge_labels=edge_label_dict, font_color=edge_label_color
                )

    def draw_separate_networks(
        self,
        figsize: tuple[int, int] = (12, 10),
        node_size: int = 500,
        font_size: int = 8,
        title: str | None = None,
        save_path: str | None = None,
        show_plot: bool = True,
        layout_func: str = "planar",
        edge_labels: bool = False,
        doc_space_ratio: float = 0.5,
    ) -> None:
        """
        Draw document and semantic networks as separate groups with connecting lines between them.

        Args:
            figsize (tuple[int, int]): Figure size as (width, height) in inches.
            node_size (int): Size of the nodes.
            edge_color (str): Color of regular edges.
            font_size (int): Font size for node labels.
            title (str | None): Title for the plot.
            save_path (str | None): Path to save the figure. If None, the figure is not saved.
            show_plot (bool): Whether to show the plot.
            layout_func (str): Layout algorithm to use for each network.
        """
        if not isinstance(self.drawing_obj, Document):
            raise ValueError("This method is only applicable for Document graphs.")

        # Use existing conversion method to get the graphs
        document_graph = self.nx_graph
        semantic_graph = self.semantic_graph

        # Find cross-connections between document and semantic network
        cross_edges = [
            (parent, child)
            for parent, children in self.drawing_obj.semantic_references.adj.items()
            for child in children
        ]

        # Setup the plot
        fig, ax = plt.subplots(figsize=figsize)

        ##
        # Scaling
        ##
        width_ratio = max(0.1, min(0.9, doc_space_ratio))  # Limit between 10% and 90%

        # Calculate centers so the total width is 4 units (-2 to +2)
        total_width = 4.0
        doc_width = total_width * width_ratio
        sem_width = total_width - doc_width

        # Set centers to balance around the origin (0,0)
        doc_center = -sem_width / 2
        sem_center = doc_width / 2
        # Scale and offset to position each network
        doc_scale = width_ratio * 1.8  # Scale factor for document graph
        sem_scale = (1.0 - width_ratio) * 1.8  # Scale factor for semantic graph

        doc_offset = np.array([doc_center, 0])
        sem_offset = np.array([sem_center, 0])

        # Get layouts
        doc_pos = self.get_layout(document_graph, layout_func)
        sem_pos = self.get_layout(semantic_graph, layout_func)
        doc_pos = {
            node: (doc_scale * pos) + doc_offset for node, pos in doc_pos.items()
        }
        sem_pos = {
            node: (sem_scale * pos) + sem_offset for node, pos in sem_pos.items()
        }

        # Combine positions for cross-connections
        all_pos = {**doc_pos, **sem_pos}

        # Prepare colors
        doc_colors, doc_legend = self.prepare_colors_for_document(
            self.drawing_obj, document_graph
        )
        sem_colors, sem_legend = self.prepare_colors_for_semantic_network(
            self.drawing_obj.semantic_network, semantic_graph
        )

        # Draw document elements
        self._draw_graph_elements(
            document_graph,
            doc_pos,
            node_size,
            doc_colors,
            self.DEFAULT_COLORS["document"]["edge"],
            font_size,
            edge_labels,
        )

        # Draw semantic elements
        self._draw_graph_elements(
            semantic_graph,
            sem_pos,
            node_size,
            sem_colors,
            self.DEFAULT_COLORS["semantic"]["edge"],
            font_size,
            edge_labels,
        )

        # Draw cross-connections
        if cross_edges:
            nx.draw_networkx_edges(
                nx.DiGraph(cross_edges),
                all_pos,
                edge_color=self.DEFAULT_COLORS["cross"]["edge"],
                style="dashed",
                width=1.0,
                alpha=0.6,
                connectionstyle="arc3,rad=0.2",
            )

        # Add section labels
        label_box = dict(facecolor="lightgray", alpha=0.5, boxstyle="round")

        doc_y_values = [pos[1] for pos in doc_pos.values()]
        sem_y_values = [pos[1] for pos in sem_pos.values()]
        section_y = max(
            max(doc_y_values) if doc_y_values else 0,
            max(sem_y_values) if sem_y_values else 0,
        )

        # Place labels above the networks with a small margin
        plt.text(
            doc_center,
            section_y + 0.3,
            "Document Structure",
            fontsize=14,
            ha="center",
            bbox=label_box,
        )
        plt.text(
            sem_center,
            section_y + 0.3,
            "Semantic Network",
            fontsize=14,
            ha="center",
            bbox=label_box,
        )

        # Ensure the labels are visible by adjusting the plot limits
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        margin = 0.1
        plt.xlim(doc_center - doc_scale - margin, sem_center + sem_scale + margin)
        plt.ylim(y_min, max(y_max, section_y + 0.6))

        # Add legend
        if doc_legend or sem_legend:
            combined_legend = {**doc_legend, **sem_legend}
            self._add_legend(
                combined_legend, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3
            )

        # Finalize and display the plot
        self._finalize_plot(title, save_path, show_plot)

    def draw(
        self,
        figsize: tuple[int, int] = (10, 8),
        node_size: int = 500,
        node_colors: list[str] | None = None,
        edge_color: str = "gray",
        font_size: int = 8,
        edge_labels: bool = False,
        title: str | None = None,
        layout_func: str = "spring",
        save_path: str | None = None,
        show_plot: bool = True,
    ) -> None:
        """
        Draw the graph using NetworkX and Matplotlib.

        Args:
            figsize (tuple[int, int]): Figure size as (width, height) in inches.
            node_size (int): Size of the nodes.
            node_colors (list[str] | None): Colors of the nodes.
            edge_color (str): Color of the edges.
            font_size (int): Font size for node labels.
            edge_labels (bool): Whether to show edge labels.
            title (str | None): Title for the plot.
            layout_func (str): Layout algorithm to use.
            save_path (str | None): Path to save the figure. If None, the figure is not saved.
            show_plot (bool): Whether to show the plot.
            group_semantic_nodes (bool): Whether to visually group semantic nodes.
        """
        plt.figure(figsize=figsize)

        # Get layout function and calculate positions
        layout_function = self._get_layout_function(layout_func)
        pos = layout_function(self.nx_graph)

        # Prepare node colors
        legend = None
        if node_colors is None:
            if isinstance(self.drawing_obj, Document):
                node_colors, legend = self.prepare_colors_for_document(
                    self.drawing_obj, self.nx_graph
                )
            elif isinstance(self.drawing_obj, SemanticKnowledgeGraph):
                node_colors, legend = self.prepare_colors_for_semantic_network(
                    self.drawing_obj, self.nx_graph
                )
            else:
                node_colors, legend = self.prepare_colors_for_reference_graph(
                    self.drawing_obj, self.nx_graph
                )

        # Use the helper method to draw nodes, edges, and labels
        self._draw_graph_elements(
            self.nx_graph,
            pos,
            node_size,
            node_colors,
            edge_color,
            font_size,
            edge_labels,
        )

        # Add legend
        if legend:
            self._add_legend(legend, loc="upper right", fontsize=8)

        # Finalize and display the plot
        self._finalize_plot(title, save_path, show_plot)

    def highlight_path(
        self,
        source: str,
        target: str,
        figsize: tuple[int, int] = (10, 8),
        show_plot: bool = True,
        save_path: str | None = None,
    ) -> bool:
        """
        Highlight a path from source to target if it exists.

        Args:
            source (str): Source node.
            target (str): Target node.
            figsize (tuple[int, int]): Figure size as (width, height) in inches.
            path_color (str): Color for the highlighted path.
            show_plot (bool): Whether to show the plot.
            save_path (Optional[str]): Path to save the figure. If None, the figure is not saved.

        Returns:
            bool: True if a path exists, False otherwise.
        """
        # Check if both nodes exist in the graph
        if source not in self.nx_graph.nodes() or target not in self.nx_graph.nodes():
            print(f"One or both nodes ({source}, {target}) not in graph.")
            return False

        try:
            # Find a path from source to target
            path = nx.shortest_path(self.nx_graph, source=source, target=target)

            # Create edges from the path
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

            # Draw the graph
            plt.figure(figsize=figsize)
            pos = nx.spring_layout(self.nx_graph)

            # Draw all nodes and edges first using helper method
            self._draw_graph_elements(
                self.nx_graph,
                pos,
                node_size=700,
                node_color="lightblue",
                edge_color="gray",
                font_size=10,
                edge_labels=False,
            )

            # Highlight the path nodes and edges separately
            nx.draw_networkx_nodes(
                self.nx_graph,
                pos,
                nodelist=path,
                node_size=700,
                node_color=self.DEFAULT_COLORS["highlight"]["node"],
                alpha=0.8,
            )
            nx.draw_networkx_edges(
                self.nx_graph,
                pos,
                edgelist=path_edges,
                width=2.5,
                edge_color=self.DEFAULT_COLORS["highlight"]["edge"],
                alpha=1.0,
            )

            self._finalize_plot(
                title=f"Path from '{source}' to '{target}'",
                save_path=save_path,
                show_plot=show_plot,
            )

            return True

        except nx.NetworkXNoPath:
            print(f"No path exists from {source} to {target}.")
            return False

    def visualize_subgraph(
        self,
        nodes: list[str],
        figsize: tuple[int, int] = (10, 8),
        title: str | None = None,
        save_path: str | None = None,
        show_plot: bool = True,
    ) -> None:
        """
        Visualize a subgraph containing only the specified nodes.

        Args:
            nodes (list[str]): list of nodes to include in the subgraph.
            figsize (tuple[int, int]): Figure size as (width, height) in inches.
            title (Optional[str]): Title for the plot.
            save_path (Optional[str]): Path to save the figure. If None, the figure is not saved.
            show_plot (bool): Whether to show the plot.
        """
        # Create a subgraph with the specified nodes
        subgraph = self.nx_graph.subgraph(nodes)

        plt.figure(figsize=figsize)
        pos = nx.spring_layout(subgraph)

        # Determine which nodes are roots in the original graph
        root_nodes = self.drawing_obj.roots
        node_colors = [
            "lightgreen" if node in root_nodes else "lightblue"
            for node in subgraph.nodes()
        ]

        # Draw the subgraph
        self._draw_graph_elements(
            subgraph,
            pos,
            node_size=1000,
            node_color=node_colors,
            edge_color="gray",
            font_size=10,
            edge_labels=False,
        )

        self._finalize_plot(
            title or f"Subgraph with {len(nodes)} nodes", save_path, show_plot
        )

    def prepare_colors_for_semantic_network(
        self, drawing_obj, graph
    ) -> tuple[list[str], dict[str, str]]:
        """
        Prepare colors for nodes in the semantic network graph.

        Returns:
            tuple[list[str], dict[str, str]]: A list of colors for the nodes in the graph and a legend.
        """
        if not isinstance(drawing_obj, SemanticKnowledgeGraph):
            raise ValueError(
                "This method is only applicable for SemanticNetwork graphs."
            )
        node_colors = {}

        color_types = {
            k: plt.cm.viridis(i / len(drawing_obj.available_types))[:3]
            for i, k in enumerate(drawing_obj.available_types)
        }

        for node in graph.nodes():
            if node in drawing_obj.entities:
                try:
                    node_colors[node] = color_types[drawing_obj.entities[node].type.uri]
                except KeyError:
                    node_colors[node] = "gray"
            elif node in color_types:
                node_colors[node] = color_types[node]

        colors = list(node_colors.values())
        legend = {
            **{color: cat for cat, color in color_types.items()},
        }
        return colors, legend

    def prepare_colors_for_document(
        self, drawing_obj, graph
    ) -> tuple[list[str], dict[str, str]]:
        """
        Prepare colors for nodes in the document graph.

        Returns:
            tuple[list[str], dict[str, str]]: A list of colors for the nodes in the graph and a legend.
        """
        if not isinstance(drawing_obj, Document):
            raise ValueError("This method is only applicable for Document graphs.")
        node_colors = {}

        cat_set = set()
        todo = set()

        for node in graph.nodes():
            if node in drawing_obj.references.roots:  # pages
                node_colors[node] = "lightgreen"
            elif node in drawing_obj.semantic_network.entities:
                node_colors[node] = "lightblue"
            elif node in drawing_obj.semantic_network.available_types:
                node_colors[node] = "orange"
            else:
                try:
                    cat_set.add(drawing_obj.get_element_obj(node).category)
                    todo.add(node)
                except KeyError:
                    node_colors[node] = "gray"
        # Assign colors to nodes based on their categories
        # make one color for each category
        cat_set = list(cat_set)
        cat_set.sort()
        color_map = {
            cat: plt.cm.viridis(i / len(cat_set))[:3] for i, cat in enumerate(cat_set)
        }
        colors = list()
        for node in graph.nodes():
            if node in todo:
                colors.append(color_map[drawing_obj.get_element_obj(node).category])
            else:
                colors.append(node_colors[node])
        legend = {
            "lightgreen": "Page",
            "lightblue": "Entity",
            "orange": "Type",
            **{color_map[cat]: cat for cat in cat_set},
        }
        return colors, legend

    def prepare_colors_for_reference_graph(
        self, drawing_obj, graph
    ) -> tuple[list[str], dict[str, str]]:
        """
        Prepare colors for nodes in the reference graph.

        Returns:
            tuple[list[str], dict[str, str]]: A list of colors for the nodes in the graph and a legend.
        """
        if not isinstance(drawing_obj, ReferenceGraph):
            raise ValueError(
                "This method is only applicable for ReferenceGraph graphs."
            )
        node_colors = {}

        # color_types = {
        #     k: plt.cm.viridis(i / len(drawing_obj.available_types))[:3]
        #     for i, k in enumerate(drawing_obj.available_types)
        # }

        for node in graph.nodes():
            if node in drawing_obj.roots:
                node_colors[node] = "lightgreen"
            else:
                node_colors[node] = "lightblue"
        return list(node_colors.values()), {
            "lightgreen": "Root",
            "lightblue": "Child",
        }
