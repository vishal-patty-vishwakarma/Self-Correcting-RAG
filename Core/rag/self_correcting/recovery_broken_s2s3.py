from typing import Any, List, Optional, Sequence, Set

from Core.Index.Tree import DocumentTree, TreeNode

from .types import RecoveryStrategy


class RecoveryExecutor:
    """
    Execute retrieval-recovery strategies against BookRAG indexes.

    Current implementation status:

        S1_LOCAL_CONTEXT       -> IMPLEMENTED using DocumentTree
        S2_ENTITY_GUIDED       -> pending complete KG
        S3_GRAPH_MULTI_HOP     -> pending complete KG
        S4_STRUCTURAL_SCOPE    -> IMPLEMENTED using DocumentTree

    Design:

        S1 = immediate structural neighborhood

        S4 = wider bounded structural neighborhood

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

        # S1:
        # previous 1 sibling + next 1 sibling
        self.local_sibling_radius = local_sibling_radius

        # S4:
        # previous 2 sibling sections + next 2 sibling sections
        self.structural_sibling_radius = structural_sibling_radius

        # S3:
        # bounded KG traversal depth
        self.graph_hops = graph_hops

        # Prevent uncontrolled graph expansion.
        self.max_graph_entities = max_graph_entities
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
        - TreeNode objects,
        - integer node IDs,
        - dictionaries containing "index_id".

        query is accepted now because S2/S3 will later need it.
        """

        node_ids = self._normalize_node_ids(current_nodes)

        if strategy == RecoveryStrategy.S1_LOCAL_CONTEXT:
            return self._local_context_expansion(node_ids)

        if strategy == RecoveryStrategy.S4_STRUCTURAL_SCOPE:
            return self._structural_scope_expansion(node_ids)

        if strategy == RecoveryStrategy.S2_ENTITY_GUIDED:
            return self._entity_guided_expansion(
                node_ids=node_ids,
                query=query,
            )

        if strategy == RecoveryStrategy.S3_GRAPH_MULTI_HOP:
            return self._graph_multi_hop_expansion(
                node_ids=node_ids,
                query=query,
            )

        raise ValueError(
            f"Unsupported recovery strategy: {strategy}"
        )

    def is_available(
        self,
        strategy: RecoveryStrategy,
    ) -> bool:
        """
        Return whether the strategy can currently be executed.

        S1/S4 require only the DocumentTree.

        S2/S3 additionally require a usable BookRAG GraphIndex.
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
    # S2 - ENTITY-GUIDED EXPANSION
    # ==================================================================

    def _entity_guided_expansion(
        self,
        node_ids: Sequence[int],
        query: str = "",
    ) -> List[TreeNode]:
        """
        S2: Entity-guided recovery using BookRAG's KG <-> Tree mapping.

        Flow:

            original Tree nodes
                +
            query-matching KG entities
                ↓
            KG entity node names
                ↓
            GraphIndex.NodeName2TreeNodes(...)
                ↓
            additional source Tree nodes

        This is BookRAG's practical GT-Link mechanism.
        """

        if not self._has_usable_graph():
            return []

        ordered_nodes: List[TreeNode] = []
        seen: Set[int] = set()

        # Preserve original recovery anchors first.
        for node_id in node_ids:
            node = self.tree.get_node_by_index_id(node_id)

            if node is not None:
                self._append_unique(
                    ordered_nodes,
                    seen,
                    node,
                )

        seed_entities = self._collect_seed_entity_node_names(
            node_ids=node_ids,
            query=query,
        )

        for entity_node_name in seed_entities:

            tree_ids = self._tree_nodes_for_entity(
                entity_node_name
            )

            for tree_id in tree_ids:

                node = self.tree.get_node_by_index_id(
                    tree_id
                )

                if node is None:
                    continue

                self._append_unique(
                    ordered_nodes,
                    seen,
                    node,
                )

                if len(ordered_nodes) >= self.max_nodes:
                    return ordered_nodes[: self.max_nodes]

        return ordered_nodes[: self.max_nodes]

    # ==================================================================
    # S3 - GRAPH / MULTI-HOP EXPANSION
    # ==================================================================

    def _graph_multi_hop_expansion(
        self,
        node_ids: Sequence[int],
        query: str = "",
    ) -> List[TreeNode]:
        """
        S3: bounded multi-hop KG expansion.

        Flow:

            original Tree nodes
                ↓
            linked/query-relevant KG entities
                ↓
            bounded BFS over GraphIndex.kg.neighbors(...)
                ↓
            expanded KG entities
                ↓
            NodeName2TreeNodes(...)
                ↓
            supporting Tree nodes

        Expansion is bounded by:

            graph_hops
            max_graph_entities
            max_nodes
        """

        if not self._has_usable_graph():
            return []

        ordered_nodes: List[TreeNode] = []
        seen_tree_nodes: Set[int] = set()

        # Preserve original evidence first.
        for node_id in node_ids:

            node = self.tree.get_node_by_index_id(
                node_id
            )

            if node is not None:
                self._append_unique(
                    ordered_nodes,
                    seen_tree_nodes,
                    node,
                )

        seed_entities = self._collect_seed_entity_node_names(
            node_ids=node_ids,
            query=query,
        )

        if not seed_entities:
            return ordered_nodes[: self.max_nodes]

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
        # Bounded breadth-first KG traversal
        # --------------------------------------------------------------

        for _ in range(self.graph_hops):

            next_frontier: List[str] = []

            for entity_node_name in frontier:

                try:
                    neighbors = list(
                        self.graph_index.kg.neighbors(
                            entity_node_name
                        )
                    )

                except Exception:
                    continue

                # Deterministic traversal order.
                neighbors = sorted(
                    neighbors,
                    key=str,
                )

                for neighbor in neighbors:

                    if neighbor in visited_entities:
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
        # KG entities -> source Tree nodes
        # --------------------------------------------------------------

        for entity_node_name in entity_order:

            tree_ids = self._tree_nodes_for_entity(
                entity_node_name
            )

            for tree_id in tree_ids:

                node = self.tree.get_node_by_index_id(
                    tree_id
                )

                if node is None:
                    continue

                self._append_unique(
                    ordered_nodes,
                    seen_tree_nodes,
                    node,
                )

                if len(ordered_nodes) >= self.max_nodes:
                    return ordered_nodes[: self.max_nodes]

        return ordered_nodes[: self.max_nodes]

    # ==================================================================
    # GRAPH HELPERS
    # ==================================================================

    def _has_usable_graph(self) -> bool:
        """
        Check whether BookRAG's GraphIndex is actually usable.
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
            return kg.number_of_nodes() > 0

        except Exception:
            return False

    def _collect_seed_entity_node_names(
        self,
        node_ids: Sequence[int],
        query: str = "",
    ) -> List[str]:
        """
        Collect KG entity node names from two sources:

        1. entities already linked to the original Tree evidence;
        2. KG entities whose names strongly match the query.

        Tree-linked entities are kept first because they are directly
        grounded in BookRAG's existing retrieval.
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
        # 1. Entities linked to current Tree evidence
        # --------------------------------------------------------------

        for tree_id in node_ids:

            linked_entities = tree2kg.get(
                tree_id,
                set(),
            )

            for entity_node_name in sorted(
                linked_entities,
                key=str,
            ):

                if entity_node_name in seen:
                    continue

                if entity_node_name not in self.graph_index.kg:
                    continue

                seen.add(entity_node_name)
                result.append(entity_node_name)

                if (
                    len(result)
                    >= self.max_graph_entities
                ):
                    return result

        # --------------------------------------------------------------
        # 2. Query-matching KG entities
        # --------------------------------------------------------------

        query_entities = self._query_matching_entities(
            query
        )

        for entity_node_name in query_entities:

            if entity_node_name in seen:
                continue

            seen.add(entity_node_name)
            result.append(entity_node_name)

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
        Lightweight zero-API entity matching.

        This is intentionally conservative. It does not replace
        BookRAG's embedding-based entity mapper.

        It is used only as an additional recovery signal.
        """

        if (
            not query
            or not self._has_usable_graph()
        ):
            return []

        query_terms = self._graph_text_terms(
            query
        )

        if not query_terms:
            return []

        scored = []

        for node_name in self.graph_index.kg.nodes:

            node_data = self.graph_index.kg.nodes[
                node_name
            ]

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

            entity_terms = self._graph_text_terms(
                f"{entity_name} {entity_type}"
            )

            if not entity_terms:
                continue

            overlap = (
                query_terms
                & entity_terms
            )

            # Require either:
            # - every entity term to occur in the query, or
            # - at least two overlapping terms.
            full_entity_match = (
                entity_terms
                and entity_terms.issubset(
                    query_terms
                )
            )

            if (
                not full_entity_match
                and len(overlap) < 2
            ):
                continue

            score = (
                len(overlap)
                / max(
                    len(entity_terms),
                    1,
                )
            )

            if full_entity_match:
                score += 1.0

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
        Resolve one KG entity node name back to source Tree nodes.

        We intentionally use NodeName2TreeNodes() rather than
        Entities2TreeNodes(), because the latter currently contains
        a non-mutating set.union() call in upstream BookRAG.
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

        result = []

        for tree_id in tree_ids:

            try:
                tree_id = int(tree_id)

            except (
                TypeError,
                ValueError,
            ):
                continue

            if tree_id not in result:
                result.append(tree_id)

        return sorted(result)

    @staticmethod
    def _graph_text_terms(
        text: str,
    ) -> Set[str]:
        """
        Normalize text for lightweight query/entity matching.
        """

        import re

        return set(
            re.findall(
                r"[a-z0-9]+",
                str(text).lower(),
            )
        )

    # ==================================================================
    # S4 - STRUCTURAL SCOPE EXPANSION
    # ==================================================================

    def _structural_scope_expansion(
        self,
        node_ids: Sequence[int],
    ) -> List[TreeNode]:
        """
        S4: Wider but bounded structural expansion.

        Instead of climbing directly to a very broad ancestor and
        retrieving its whole subtree, S4 uses a section-oriented scope.

        For each current evidence node:

            identify structural anchor
                    ↓
            anchor section/subtree
                    +
            parent title
                    +
            nearby sibling sections/subtrees

        Default sibling radius:

            previous 2 sections
            next 2 sections

        This makes S4 broader than S1 without accidentally retrieving
        most of the document.
        """

        ordered_nodes: List[TreeNode] = []
        seen: Set[int] = set()

        for node_id in node_ids:

            node = self.tree.get_node_by_index_id(node_id)

            if node is None:
                continue

            # ----------------------------------------------------------
            # Determine a useful structural anchor.
            #
            # If the retrieved node is already a section/title-like node
            # with children, keep it.
            #
            # If it is a leaf text/table node, use its parent section.
            # ----------------------------------------------------------

            anchor = self._get_structural_anchor(node)

            # ----------------------------------------------------------
            # 1. Preserve original evidence first.
            # ----------------------------------------------------------

            self._append_unique(
                ordered_nodes,
                seen,
                node,
            )

            # ----------------------------------------------------------
            # 2. Add the structural anchor.
            # ----------------------------------------------------------

            self._append_unique(
                ordered_nodes,
                seen,
                anchor,
            )

            # ----------------------------------------------------------
            # 3. Add anchor's parent title/context.
            # ----------------------------------------------------------

            if anchor.parent is not None:

                self._append_unique(
                    ordered_nodes,
                    seen,
                    anchor.parent,
                )

            # ----------------------------------------------------------
            # 4. Add the current anchor subtree.
            #
            # This is the highest-priority structural evidence.
            # ----------------------------------------------------------

            self._append_subtree(
                root=anchor,
                output=ordered_nodes,
                seen=seen,
            )

            if len(ordered_nodes) >= self.max_nodes:
                break

            # ----------------------------------------------------------
            # 5. Add nearby sibling sections and their subtrees.
            #
            # Only bounded adjacent structural sections are used.
            # ----------------------------------------------------------

            nearby_sections = self._get_adjacent_siblings(
                anchor,
                radius=self.structural_sibling_radius,
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

                if len(ordered_nodes) >= self.max_nodes:
                    break

            if len(ordered_nodes) >= self.max_nodes:
                break

        return ordered_nodes[: self.max_nodes]

    # ==================================================================
    # STRUCTURAL HELPERS
    # ==================================================================

    @staticmethod
    def _get_structural_anchor(
        node: TreeNode,
    ) -> TreeNode:
        """
        Pick the most useful section-level anchor.

        Nodes with children are already structural containers.

        Leaf nodes such as TEXT/TABLE nodes normally use their parent
        title/section as the structural anchor.
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
        Return bounded previous/next siblings in document order.

        Example:

            parent children:
                30, 33, 35, 37, 41, 44

            node:
                37

            radius = 1:
                35, 41

            radius = 2:
                33, 35, 41, 44

        The current node itself is not returned.
        """

        if node.parent is None:
            return []

        siblings = list(node.parent.children)

        # Preserve deterministic document order.
        siblings = sorted(
            siblings,
            key=lambda n: n.index_id,
        )

        current_position: Optional[int] = None

        for index, sibling in enumerate(siblings):
            if sibling.index_id == node.index_id:
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

        return previous_nodes + next_nodes

    def _append_subtree(
        self,
        root: TreeNode,
        output: List[TreeNode],
        seen: Set[int],
    ) -> None:
        """
        Append a subtree while respecting max_nodes.

        Root may already exist in output; duplicate protection handles it.
        """

        subtree_nodes = self.tree.get_subtree_nodes(
            root.index_id
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

            if len(output) >= self.max_nodes:
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

        seen.add(node.index_id)
        output.append(node)

    @staticmethod
    def _normalize_node_ids(
        nodes: Sequence[Any],
    ) -> List[int]:
        """
        Convert supported evidence-node representations into BookRAG
        integer tree IDs.
        """

        result: List[int] = []
        seen: Set[int] = set()

        for node in nodes:

            node_id = None

            if isinstance(node, int):
                node_id = node

            elif isinstance(node, dict):
                node_id = node.get("index_id")

            else:
                node_id = getattr(
                    node,
                    "index_id",
                    None,
                )

            if node_id is None:
                continue

            try:
                node_id = int(node_id)

            except (TypeError, ValueError):
                continue

            if node_id not in seen:
                result.append(node_id)
                seen.add(node_id)

        return result