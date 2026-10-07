import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Core.configs.system_config import load_system_config
from Core.Index.GBCIndex import GBC


CONFIG = "config/baseline_kg_fallback.yaml"


def main():

    print("=" * 70)
    print("REBUILD BOOKRAG ENTITY VDB")
    print("=" * 70)

    cfg = load_system_config(CONFIG)

    # Load the already-completed Tree + Graph.
    gbc = GBC.load_gbc_index(cfg)

    print()
    print(
        "Graph nodes:",
        gbc.GraphIndex.kg.number_of_nodes(),
    )

    print(
        "Graph edges:",
        gbc.GraphIndex.kg.number_of_edges(),
    )

    print(
        "Entity VDB count BEFORE:",
        gbc.entity_vdb.collection.count(),
    )

    print()
    print("Rebuilding entity VDB from saved GraphIndex...")

    # Uses BookRAG's own implementation.
    # This resets only the entity VDB collection and recreates it
    # from the already-saved knowledge graph.
    gbc.rebuild_vdb()

    print()
    print(
        "Entity VDB count AFTER:",
        gbc.entity_vdb.collection.count(),
    )

    print()
    print("=" * 70)

    if gbc.entity_vdb.collection.count() == (
        gbc.GraphIndex.kg.number_of_nodes()
    ):
        print("SUCCESS: Graph and entity VDB counts match.")
    else:
        print("WARNING: Counts do not match.")

    print("=" * 70)


if __name__ == "__main__":
    main()