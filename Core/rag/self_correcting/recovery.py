import re
from typing import Any, List, Optional, Sequence, Set

from Core.Index.Tree import DocumentTree, TreeNode

from .types import RecoveryStrategy


class RecoveryExecutor:
    """
    Execute retrieval-recovery strategies against BookRAG indexes.

    Status:

        S1_LOCAL_CONTEXT       -> DocumentTree
        S2_ENTITY_GUIDED       -> GraphIndex / GT-Link
        S3_GRAPH_MULTI_HOP     -> GraphIndex KG traversal
        S4_STRUCTURAL_SCOPE    -> DocumentTree

    The executor performs recovery retrieval only.

    It does NOT:
    - assess evidence,
    - choose strategies,
    - generate answers,
    - call an LLM.
    """

    def __init__(
        self,
        tree: DocumentTree,
        graph_index: Any = None,
        max_nodes: int = 25,
        local_sibling_radius: int = 1,
        structural_sibling_radius: int = 2,
        graph_hops: int = 2,
        max_graph_entities: int = 20,
    ):
        self.tree = tree
        self.graph_index = graph_index
        self.max_nodes = max_nodes

        # S1
        self.local_sibling_radius = (
            local_sibling_radius
        )

        # S4
        self.structural_sibling_radius = (
            structural_sibling_radius
        )

        # S3
        self.graph_hops = graph_hops
        self.max_graph_entities = (
            max_graph_entities
        )

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def execute(
        self,
        strategy: RecoveryStrategy,
        current_nodes: Sequence[Any],
        query: str = "",
    ) -> List[TreeNode]:
        """
        Execute one recovery strategy.

        current_nodes may contain:
        - TreeNode objects
        - integer node IDs
        - dictionaries containing "index_id"
        """

        node_ids = self._normalize_node_ids(
            current_nodes
        )

        if (
            strategy
            == RecoveryStrategy.S1_LOCAL_CONTEXT
        ):
            return self._local_context_expansion(
                node_ids
            )

        if (
            strategy
            == RecoveryStrategy.S2_ENTITY_GUIDED
        ):
            return self._entity_guided_expansion(
                node_ids=node_ids,
                query=query,
            )

        if (
            strategy
            == RecoveryStrategy.S3_GRAPH_MULTI_HOP
        ):
            return self._graph_multi_hop_expansion(
                node_ids=node_ids,
                query=query,
            )

        if (
            strategy
            == RecoveryStrategy.S4_STRUCTURAL_SCOPE
        ):
            return (
                self._structural_scope_expansion(
                    node_ids
                )
            )

        raise ValueError(
            f"Unsupported recovery strategy: "
            f"{strategy}"
        )

    def is_available(
        self,
        strategy: RecoveryStrategy,
    ) -> bool:
        """
        Return whether a strategy can currently
        be executed.
        """

        if strategy in {
            RecoveryStrategy.S1_LOCAL_CONTEXT,
            RecoveryStrategy.S4_STRUCTURAL_SCOPE,
        }:
            return True

        if strategy in {
            RecoveryStrategy.S2_ENTITY_GUIDED,
            RecoveryStrategy.S3_GRAPH_MULTI_HOP,
        }:
            return self._has_usable_graph()

        return False

    # ==================================================================
    # S1 - LOCAL CONTEXT EXPANSION
    # ==================================================================

    def _local_context_expansion(
        self,
        node_ids: Sequence[int],
    ) -> List[TreeNode]:
        """
        S1: Immediate local structural expansion.

        For every current evidence node:

        - current node
        - parent
        - direct children
        - previous sibling
        - next sibling
        """

        ordered_nodes: List[TreeNode] = []
        seen: Set[int] = set()

        for node_id in node_ids:

            node = (
                self.tree
                .get_node_by_index_id(
                    node_id
                )
            )

            if node is None:
                continue

            # Original evidence.
            self._append_unique(
                ordered_nodes,
                seen,
                node,
            )

            # Parent.
            if node.parent is not None:
                self._append_unique(
                    ordered_nodes,
                    seen,
                    node.parent,
                )

            # Direct children.
            for child in node.children:

                self._append_unique(
                    ordered_nodes,
                    seen,
                    child,
                )

                if (
                    len(ordered_nodes)
                    >= self.max_nodes
                ):
                    break

            if (
                len(ordered_nodes)
                >= self.max_nodes
            ):
                break

            # Adjacent siblings only.
            adjacent_siblings = (
                self._get_adjacent_siblings(
                    node,
                    radius=(
                        self.local_sibling_radius
                    ),
                )
            )

            for sibling in adjacent_siblings:

                self._append_unique(
                    ordered_nodes,
                    seen,
                    sibling,
                )

                if (
                    len(ordered_nodes)
                    >= self.max_nodes
                ):
                    break

            if (
                len(ordered_nodes)
                >= self.max_nodes
            ):
                break

        return ordered_nodes[
            : self.max_nodes
        ]

    # ==================================================================
    # S2 - ENTITY-GUIDED EXPANSION
    # ==================================================================

    def _entity_guided_expansion(
        self,
        node_ids: Sequence[int],
        query: str = "",
    ) -> List[TreeNode]:
        """
        S2: Entity-guided recovery.

        Tree evidence
            -> linked/query-relevant KG entities
            -> NodeName2TreeNodes()
            -> additional source Tree nodes

        BookRAG's tree2kg/source_ids mapping is
        the practical GT-Link mechanism here.
        """

        if not self._has_usable_graph():
            return []

        ordered_nodes: List[TreeNode] = []
        seen: Set[int] = set()

        # Preserve original anchors first.
        for node_id in node_ids:

            node = (
                self.tree
                .get_node_by_index_id(
                    node_id
                )
            )

            if node is not None:
                self._append_unique(
                    ordered_nodes,
                    seen,
                    node,
                )

        seed_entities = (
            self._collect_seed_entity_node_names(
                node_ids=node_ids,
                query=query,
            )
        )

        for entity_node_name in seed_entities:

            tree_ids = (
                self._tree_nodes_for_entity(
                    entity_node_name
                )
            )

            for tree_id in tree_ids:

                node = (
                    self.tree
                    .get_node_by_index_id(
                        tree_id
                    )
                )

                if node is None:
                    continue

                self._append_unique(
                    ordered_nodes,
                    seen,
                    node,
                )

                if (
                    len(ordered_nodes)
                    >= self.max_nodes
                ):
                    return ordered_nodes[
                        : self.max_nodes
                    ]

        return ordered_nodes[
            : self.max_nodes
        ]

    # ==================================================================
    # S3 - GRAPH / MULTI-HOP EXPANSION
    # ==================================================================

    def _graph_multi_hop_expansion(
        self,
        node_ids: Sequence[int],
        query: str = "",
    ) -> List[TreeNode]:
        """
        S3: Bounded multi-hop KG expansion.

        Tree evidence
            -> seed KG entities
            -> bounded BFS over KG neighbors
            -> expanded entities
            -> NodeName2TreeNodes()
            -> supporting Tree evidence
        """

        if not self._has_usable_graph():
            return []

        ordered_nodes: List[TreeNode] = []
        seen_tree_nodes: Set[int] = set()

        # Preserve original evidence.
        for node_id in node_ids:

            node = (
                self.tree
                .get_node_by_index_id(
                    node_id
                )
            )

            if node is not None:
                self._append_unique(
                    ordered_nodes,
                    seen_tree_nodes,
                    node,
                )

        seed_entities = (
            self._collect_seed_entity_node_names(
                node_ids=node_ids,
                query=query,
            )
        )

        if not seed_entities:
            return ordered_nodes[
                : self.max_nodes
            ]

        visited_entities: Set[str] = set(
            seed_entities
        )

        entity_order: List[str] = list(
            seed_entities
        )

        frontier: List[str] = list(
            seed_entities
        )

        # --------------------------------------------------------------
        # Bounded breadth-first graph traversal.
        # --------------------------------------------------------------

        for _ in range(self.graph_hops):

            next_frontier: List[str] = []

            for entity_node_name in frontier:

                try:
                    neighbors = list(
                        self.graph_index
                        .kg
                        .neighbors(
                            entity_node_name
                        )
                    )

                except Exception:
                    continue

                # Deterministic ordering.
                neighbors = sorted(
                    neighbors,
                    key=str,
                )

                for neighbor in neighbors:

                    if (
                        neighbor
                        in visited_entities
                    ):
                        continue

                    visited_entities.add(
                        neighbor
                    )

                    entity_order.append(
                        neighbor
                    )

                    next_frontier.append(
                        neighbor
                    )

                    if (
                        len(entity_order)
                        >= self.max_graph_entities
                    ):
                        break

                if (
                    len(entity_order)
                    >= self.max_graph_entities
                ):
                    break

            frontier = next_frontier

            if not frontier:
                break

            if (
                len(entity_order)
                >= self.max_graph_entities
            ):
                break

        # --------------------------------------------------------------
        # Expanded KG entities -> Tree source nodes.
        # --------------------------------------------------------------

        for entity_node_name in entity_order:

            tree_ids = (
                self._tree_nodes_for_entity(
                    entity_node_name
                )
            )

            for tree_id in tree_ids:

                node = (
                    self.tree
                    .get_node_by_index_id(
                        tree_id
                    )
                )

                if node is None:
                    continue

                self._append_unique(
                    ordered_nodes,
                    seen_tree_nodes,
                    node,
                )

                if (
                    len(ordered_nodes)
                    >= self.max_nodes
                ):
                    return ordered_nodes[
                        : self.max_nodes
                    ]

        return ordered_nodes[
            : self.max_nodes
        ]

    # ==================================================================
    # S4 - STRUCTURAL SCOPE EXPANSION
    # ==================================================================

    def _structural_scope_expansion(
        self,
        node_ids: Sequence[int],
    ) -> List[TreeNode]:
        """
        S4: Wider but bounded structural
        expansion.

        For each evidence node:

        original node
            +
        structural anchor
            +
        anchor parent
            +
        anchor subtree
            +
        nearby sibling sections/subtrees
        """

        ordered_nodes: List[TreeNode] = []
        seen: Set[int] = set()

        for node_id in node_ids:

            node = (
                self.tree
                .get_node_by_index_id(
                    node_id
                )
            )

            if node is None:
                continue

            anchor = (
                self._get_structural_anchor(
                    node
                )
            )

            # Original evidence.
            self._append_unique(
                ordered_nodes,
                seen,
                node,
            )

            # Structural anchor.
            self._append_unique(
                ordered_nodes,
                seen,
                anchor,
            )

            # Anchor's parent.
            if anchor.parent is not None:

                self._append_unique(
                    ordered_nodes,
                    seen,
                    anchor.parent,
                )

            # Anchor subtree.
            self._append_subtree(
                root=anchor,
                output=ordered_nodes,
                seen=seen,
            )

            if (
                len(ordered_nodes)
                >= self.max_nodes
            ):
                break

            # Nearby structural sections.
            nearby_sections = (
                self._get_adjacent_siblings(
                    anchor,
                    radius=(
                        self
                        .structural_sibling_radius
                    ),
                )
            )

            for section in nearby_sections:

                self._append_unique(
                    ordered_nodes,
                    seen,
                    section,
                )

                self._append_subtree(
                    root=section,
                    output=ordered_nodes,
                    seen=seen,
                )

                if (
                    len(ordered_nodes)
                    >= self.max_nodes
                ):
                    break

            if (
                len(ordered_nodes)
                >= self.max_nodes
            ):
                break

        return ordered_nodes[
            : self.max_nodes
        ]

    # ==================================================================
    # GRAPH HELPERS
    # ==================================================================

    def _has_usable_graph(
        self,
    ) -> bool:
        """
        Check whether BookRAG GraphIndex is
        available and non-empty.
        """

        if self.graph_index is None:
            return False

        kg = getattr(
            self.graph_index,
            "kg",
            None,
        )

        if kg is None:
            return False

        if not hasattr(
            self.graph_index,
            "NodeName2TreeNodes",
        ):
            return False

        try:
            return (
                kg.number_of_nodes()
                > 0
            )

        except Exception:
            return False

    def _collect_seed_entity_node_names(
        self,
        node_ids: Sequence[int],
        query: str = "",
    ) -> List[str]:
        """
        Collect graph seed entities from:

        1. entities linked to the original
           Tree evidence;
        2. entities whose names match the query.

        Linked evidence entities are kept first.
        """

        if not self._has_usable_graph():
            return []

        result: List[str] = []
        seen: Set[str] = set()

        tree2kg = getattr(
            self.graph_index,
            "tree2kg",
            {},
        )

        # --------------------------------------------------------------
        # 1. KG entities linked to current Tree evidence.
        # --------------------------------------------------------------

        for tree_id in node_ids:

            linked_entities = (
                tree2kg.get(
                    tree_id,
                    set(),
                )
            )

            for entity_node_name in sorted(
                linked_entities,
                key=str,
            ):

                if (
                    entity_node_name
                    in seen
                ):
                    continue

                if (
                    entity_node_name
                    not in self.graph_index.kg
                ):
                    continue

                seen.add(
                    entity_node_name
                )

                result.append(
                    entity_node_name
                )

                if (
                    len(result)
                    >= self.max_graph_entities
                ):
                    return result

        # --------------------------------------------------------------
        # 2. Query-matching KG entities.
        # --------------------------------------------------------------

        query_entities = (
            self._query_matching_entities(
                query
            )
        )

        for entity_node_name in query_entities:

            if entity_node_name in seen:
                continue

            seen.add(
                entity_node_name
            )

            result.append(
                entity_node_name
            )

            if (
                len(result)
                >= self.max_graph_entities
            ):
                break

        return result

    def _query_matching_entities(
        self,
        query: str,
    ) -> List[str]:
        """
        Lightweight zero-API query/entity matching.

        This is only a recovery signal.
        It does not replace BookRAG's normal
        embedding-based entity mapper.
        """

        if (
            not query
            or not self._has_usable_graph()
        ):
            return []

        query_terms = (
            self._graph_text_terms(
                query
            )
        )

        if not query_terms:
            return []

        scored = []

        for node_name in (
            self.graph_index.kg.nodes
        ):

            node_data = (
                self.graph_index
                .kg
                .nodes[node_name]
            )

            entity_name = str(
                node_data.get(
                    "entity_name",
                    node_name,
                )
            )

            entity_type = str(
                node_data.get(
                    "entity_type",
                    "",
                )
            )

            # Match primarily on entity name.
            entity_name_terms = (
                self._graph_text_terms(
                    entity_name
                )
            )

            entity_type_terms = (
                self._graph_text_terms(
                    entity_type
                )
            )

            if not entity_name_terms:
                continue

            overlap = (
                query_terms
                & entity_name_terms
            )

            full_name_match = (
                entity_name_terms
                .issubset(
                    query_terms
                )
            )

            # Multi-token entity:
            # require at least two matched words,
            # unless the full name matches.
            #
            # Single-token entity:
            # require the full token to match.
            if len(entity_name_terms) == 1:

                if not full_name_match:
                    continue

            else:

                if (
                    not full_name_match
                    and len(overlap) < 2
                ):
                    continue

            score = (
                len(overlap)
                / max(
                    len(entity_name_terms),
                    1,
                )
            )

            if full_name_match:
                score += 1.0

            # Tiny deterministic tie-break signal
            # when the type is also present.
            type_overlap = (
                query_terms
                & entity_type_terms
            )

            if type_overlap:
                score += 0.05

            scored.append(
                (
                    score,
                    str(node_name),
                )
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        return [
            node_name
            for _, node_name in scored[
                : self.max_graph_entities
            ]
        ]

    def _tree_nodes_for_entity(
        self,
        entity_node_name: str,
    ) -> List[int]:
        """
        Map one KG entity back to BookRAG Tree
        source nodes.

        NodeName2TreeNodes() is used deliberately.
        """

        try:
            tree_ids = (
                self.graph_index
                .NodeName2TreeNodes(
                    entity_node_name
                )
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            return []

        result: List[int] = []

        for tree_id in tree_ids:

            try:
                tree_id = int(
                    tree_id
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            if tree_id not in result:
                result.append(
                    tree_id
                )

        return sorted(
            result
        )

    @staticmethod
    def _graph_text_terms(
        text: str,
    ) -> Set[str]:
        """
        Normalize text for lightweight
        query/entity matching.
        """

        return set(
            re.findall(
                r"[a-z0-9]+",
                str(text).lower(),
            )
        )

    # ==================================================================
    # STRUCTURAL HELPERS
    # ==================================================================

    @staticmethod
    def _get_structural_anchor(
        node: TreeNode,
    ) -> TreeNode:
        """
        Pick a useful section-level anchor.

        Nodes with children are structural
        containers.

        Leaf nodes use their parent section
        when available.
        """

        if node.children:
            return node

        if node.parent is not None:
            return node.parent

        return node

    @staticmethod
    def _get_adjacent_siblings(
        node: TreeNode,
        radius: int,
    ) -> List[TreeNode]:
        """
        Return bounded previous/next siblings
        in document order.
        """

        if node.parent is None:
            return []

        siblings = list(
            node.parent.children
        )

        siblings = sorted(
            siblings,
            key=lambda n: n.index_id,
        )

        current_position: Optional[int] = None

        for index, sibling in enumerate(
            siblings
        ):
            if (
                sibling.index_id
                == node.index_id
            ):
                current_position = index
                break

        if current_position is None:
            return []

        previous_nodes = siblings[
            max(
                0,
                current_position - radius,
            ):
            current_position
        ]

        next_nodes = siblings[
            current_position + 1:
            current_position + 1 + radius
        ]

        return (
            previous_nodes
            + next_nodes
        )

    def _append_subtree(
        self,
        root: TreeNode,
        output: List[TreeNode],
        seen: Set[int],
    ) -> None:
        """
        Append a subtree while respecting
        max_nodes.
        """

        subtree_nodes = (
            self.tree.get_subtree_nodes(
                root.index_id
            )
        )

        subtree_nodes = sorted(
            subtree_nodes,
            key=lambda n: n.index_id,
        )

        for subtree_node in subtree_nodes:

            self._append_unique(
                output,
                seen,
                subtree_node,
            )

            if (
                len(output)
                >= self.max_nodes
            ):
                return

    # ==================================================================
    # GENERIC HELPERS
    # ==================================================================

    @staticmethod
    def _append_unique(
        output: List[TreeNode],
        seen: Set[int],
        node: TreeNode,
    ) -> None:
        """
        Append a TreeNode only once.
        """

        if node is None:
            return

        if node.index_id in seen:
            return

        seen.add(
            node.index_id
        )

        output.append(
            node
        )

    @staticmethod
    def _normalize_node_ids(
        nodes: Sequence[Any],
    ) -> List[int]:
        """
        Convert supported evidence representations
        into BookRAG integer Tree IDs.
        """

        result: List[int] = []
        seen: Set[int] = set()

        for node in nodes:

            node_id = None

            if isinstance(
                node,
                int,
            ):
                node_id = node

            elif isinstance(
                node,
                dict,
            ):
                node_id = node.get(
                    "index_id"
                )

            else:
                node_id = getattr(
                    node,
                    "index_id",
                    None,
                )

            if node_id is None:
                continue

            try:
                node_id = int(
                    node_id
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            if node_id not in seen:

                result.append(
                    node_id
                )

                seen.add(
                    node_id
                )

        return result