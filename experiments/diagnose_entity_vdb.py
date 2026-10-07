import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Core.configs.system_config import load_system_config
from Core.Index.GBCIndex import GBC


CONFIG = "config/baseline_kg_fallback.yaml"


def main():

    cfg = load_system_config(CONFIG)

    gbc = GBC.load_gbc_index(cfg)

    print("=" * 70)
    print("ENTITY VDB DIAGNOSTIC")
    print("=" * 70)

    print(
        "Collection count:",
        gbc.entity_vdb.collection.count(),
    )

    queries = [
        "strainer",
        "pump",
        "Reactor R-2",
        (
            "What strainer protects the pump "
            "that serves Reactor R-2?"
        ),
    ]

    for query in queries:

        print()
        print("QUERY:", query)

        results = gbc.entity_vdb.search(
            query_text=query,
            top_k=5,
        )

        print(
            "RESULT COUNT:",
            len(results),
        )

        for result in results[:3]:

            print(
                "  distance:",
                result.get("distance"),
            )

            print(
                "  entity:",
                result.get(
                    "metadata",
                    {},
                ).get("entity_name"),
            )

            print(
                "  type:",
                result.get(
                    "metadata",
                    {},
                ).get("entity_type"),
            )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()