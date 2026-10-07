import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Core.Index.Tree import DocumentTree
from Core.Index.Graph import Graph

from Core.rag.self_correcting import (
    RecoveryExecutor,
    RecoveryStrategy,
)


# Workspace root that holds this repository (…/BookRAG/BookRAG).
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = str(
    WORKSPACE_ROOT / "test_output" / "controlled_manual"
)

TREE_PATH = str(
    WORKSPACE_ROOT / "test_output" / "controlled_manual" / "tree.pkl"
)


def main():

    print("=" * 80)
    print("REAL BOOKRAG GRAPH VALIDATION")
    print("=" * 80)

    # --------------------------------------------------------------
    # Load REAL indexes
    # --------------------------------------------------------------

    tree = DocumentTree.load_from_file(
        TREE_PATH
    )

    graph = Graph.load_from_dir(
        OUTPUT_DIR
    )

    print()
    print(
        "KG nodes:",
        graph.kg.number_of_nodes(),
    )

    print(
        "KG edges:",
        graph.kg.number_of_edges(),
    )

    print(
        "Tree nodes with KG links:",
        len(graph.tree2kg),
    )

    # --------------------------------------------------------------
    # Inspect controlled cooling-system region
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print("TREE -> KG LINKS")
    print("-" * 80)

    for tree_id in [
        37,
        38,
        39,
        40,
        41,
        42,
        43,
    ]:

        linked = graph.tree2kg.get(
            tree_id,
            set(),
        )

        print(
            f"Tree {tree_id}:",
            sorted(linked),
        )

    # --------------------------------------------------------------
    # Recovery executor using REAL BookRAG KG
    # --------------------------------------------------------------

    recovery = RecoveryExecutor(
        tree=tree,
        graph_index=graph,
        max_nodes=25,
        graph_hops=2,
    )

    print()
    print("-" * 80)
    print("STRATEGY AVAILABILITY")
    print("-" * 80)

    print(
        "S2 available:",
        recovery.is_available(
            RecoveryStrategy.S2_ENTITY_GUIDED
        ),
    )

    print(
        "S3 available:",
        recovery.is_available(
            RecoveryStrategy.S3_GRAPH_MULTI_HOP
        ),
    )

    # --------------------------------------------------------------
    # REAL S2
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print("REAL S2 — ENTITY-GUIDED RECOVERY")
    print("=" * 80)

    s2_nodes = recovery.execute(
        RecoveryStrategy.S2_ENTITY_GUIDED,
        [37],
        query="Where is Pump Y mentioned?",
    )

    s2_ids = [
        node.index_id
        for node in s2_nodes
    ]

    print(
        "S2 recovered Tree IDs:",
        s2_ids,
    )

    # --------------------------------------------------------------
    # REAL S3
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print("REAL S3 — GRAPH MULTI-HOP RECOVERY")
    print("=" * 80)

    s3_nodes = recovery.execute(
        RecoveryStrategy.S3_GRAPH_MULTI_HOP,
        [37],
        query=(
            "What strainer protects the pump "
            "that serves Reactor R-2?"
        ),
    )

    s3_ids = [
        node.index_id
        for node in s3_nodes
    ]

    print(
        "S3 recovered Tree IDs:",
        s3_ids,
    )

    print()
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()