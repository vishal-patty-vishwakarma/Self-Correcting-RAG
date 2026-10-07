import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Core.configs.system_config import load_system_config
from Core.configs.llm_config import LLMConfig
from Core.configs.vlm_config import VLMConfig
from Core.configs.rag.gbc_config import GBCRAGConfig

from Core.Index.GBCIndex import GBC

from Core.rag import create_rag_agent
from Core.rag.gbc_utils import SubStep

from Core.rag.self_correcting.eam import EvidenceAssessmentModule
from Core.rag.self_correcting.controller import RetryController
from Core.rag.self_correcting.types import RetryState


# ======================================================================
# CONFIG
# ======================================================================

CONFIG = "config/baseline_kg_fallback.yaml"

OUTPUT_FILE = (
    PROJECT_ROOT.parent.parent
    / "test_output"
    / "controlled_manual"
    / "candidate_screen"
    / "retrieval_candidate_screen.json"
)


# ======================================================================
# CANDIDATE QUERIES
#
# These use facts already established in the controlled manual.
#
# We are NOT judging final answer quality here.
#
# We are only asking:
#
#   Does normal baseline retrieval produce evidence that EAM considers
#   incomplete enough to trigger recovery?
# ======================================================================

CANDIDATES = [
    {
        "name": "multihop_strainer_direct",
        "query": (
            "What strainer protects the pump "
            "that serves Reactor R-2?"
        ),
        "expected": "ST-C40",
    },
    {
        "name": "multihop_strainer_indirect",
        "query": (
            "Which inlet protection device belongs to the "
            "circulation equipment serving Reactor R-2?"
        ),
        "expected": "ST-C40",
    },
    {
        "name": "pump_alias_and_strainer",
        "query": (
            "For the circulation equipment serving Reactor R-2, "
            "what is its alternate identifier and which inlet "
            "strainer protects it?"
        ),
        "expected": "PX-40 / Pump Y + ST-C40",
    },
    {
        "name": "indirect_upstream_component",
        "query": (
            "What protective component is upstream of the equipment "
            "responsible for circulating coolant to Reactor R-2, "
            "and what is that equipment's identifier?"
        ),
        "expected": "ST-C40 + PX-40 / Pump Y",
    },
    {
        "name": "current_interval",
        "query": (
            "What is the current inspection interval "
            "for Valve X?"
        ),
        "expected": "500 operating hours",
    },
    {
        "name": "supersession_plus_current_value",
        "query": (
            "What is the current inspection interval for Valve X, "
            "and which service bulletin supersedes the older "
            "maintenance guidance?"
        ),
        "expected": "500 operating hours + SB-17",
    },
    {
        "name": "controlled_absence_voltage",
        "query": (
            "What is the operating voltage of Valve X?"
        ),
        "expected": "Not specified",
    },
    {
        "name": "current_and_legacy_actuator",
        "query": (
            "What is the current actuator for Valve X, "
            "and which legacy actuator is no longer preferred?"
        ),
        "expected": "AX-4 + AX-3",
    },
]


# ======================================================================
# HELPERS
# ======================================================================


def find_typed_config(obj, target_type, max_depth=5):
    """
    Find an instantiated config object inside SystemConfig
    without depending on the exact top-level attribute name.
    """

    visited = set()

    def walk(value, depth):
        if value is None:
            return None

        if isinstance(value, target_type):
            return value

        if depth > max_depth:
            return None

        value_id = id(value)

        if value_id in visited:
            return None

        visited.add(value_id)

        if isinstance(value, dict):
            children = value.values()

        elif isinstance(value, (list, tuple, set)):
            children = value

        else:
            try:
                children = vars(value).values()
            except TypeError:
                return None

        for child in children:
            found = walk(
                child,
                depth + 1,
            )

            if found is not None:
                return found

        return None

    return walk(obj, 0)


def node_ids_from_dicts(nodes):
    """
    Extract integer Tree IDs from BookRAG retrieval-node dictionaries.
    """

    ids = []

    for node in nodes:
        if not isinstance(node, dict):
            continue

        node_id = node.get("index_id")

        if node_id is None:
            continue

        try:
            node_id = int(node_id)

        except (TypeError, ValueError):
            continue

        ids.append(node_id)

    return ids


def enum_value(value):
    """
    Return Enum.value when available, otherwise return the object.
    """

    if value is None:
        return None

    return getattr(
        value,
        "value",
        value,
    )


# ======================================================================
# MAIN
# ======================================================================


def main():

    print("=" * 90)
    print("BOOKRAG BASELINE RETRIEVAL -> EAM CANDIDATE SCREEN")
    print("=" * 90)

    print()
    print("Config:")
    print(CONFIG)

    print()
    print(
        "Candidate count:",
        len(CANDIDATES),
    )

    # ------------------------------------------------------------------
    # 1. Load BookRAG once.
    # ------------------------------------------------------------------

    print()
    print("=" * 90)
    print("LOADING BOOKRAG")
    print("=" * 90)

    system_cfg = load_system_config(
        CONFIG
    )

    gbc = GBC.load_gbc_index(
        system_cfg
    )

    rag_cfg = find_typed_config(
        system_cfg,
        GBCRAGConfig,
    )

    llm_cfg = find_typed_config(
        system_cfg,
        LLMConfig,
    )

    vlm_cfg = find_typed_config(
        system_cfg,
        VLMConfig,
    )

    if rag_cfg is None:
        raise RuntimeError(
            "Could not locate GBCRAGConfig."
        )

    if llm_cfg is None:
        raise RuntimeError(
            "Could not locate LLMConfig."
        )

    if vlm_cfg is None:
        raise RuntimeError(
            "Could not locate VLMConfig."
        )

    rag = create_rag_agent(
        strategy_config=rag_cfg,
        llm_config=llm_cfg,
        vlm_config=vlm_cfg,
        gbc_index=gbc,
    )

    print()
    print(
        "RAG type:",
        type(rag).__name__,
    )

    print(
        "Self-correcting enabled in retrieval config:",
        getattr(
            rag,
            "self_correcting_enabled",
            None,
        ),
    )

    # ------------------------------------------------------------------
    # 2. Separate EAM + controller used only for screening.
    # ------------------------------------------------------------------

    eam = EvidenceAssessmentModule()

    controller = RetryController()

    results = []

    # ------------------------------------------------------------------
    # 3. Screen every candidate.
    # ------------------------------------------------------------------

    for number, candidate in enumerate(
        CANDIDATES,
        start=1,
    ):

        query = candidate["query"]

        print()
        print()
        print("=" * 90)
        print(
            f"CANDIDATE {number}/{len(CANDIDATES)}"
        )
        print("=" * 90)

        print(
            "Name:",
            candidate["name"],
        )

        print(
            "Query:",
            query,
        )

        print(
            "Expected:",
            candidate["expected"],
        )

        current_step = SubStep(
            sub_query=query,
            sub_number=number,
        )

        # --------------------------------------------------------------
        # Baseline retrieval only.
        #
        # IMPORTANT:
        # We intentionally do NOT call:
        #
        #     rag.answer.answer_simple_question(...)
        #
        # and we intentionally do NOT call:
        #
        #     rag._apply_self_correcting_recovery(...)
        #
        # This gives us the normal baseline retrieval evidence only.
        # --------------------------------------------------------------

        retrieval_start = time.perf_counter()

        try:

            rag._retrieve(
                query,
                current_step,
            )

        except Exception as exc:

            retrieval_seconds = (
                time.perf_counter()
                - retrieval_start
            )

            print()
            print("RETRIEVAL ERROR:")
            print(
                type(exc).__name__,
                str(exc),
            )

            results.append(
                {
                    "name": candidate["name"],
                    "query": query,
                    "expected": candidate["expected"],
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "retrieval_seconds": retrieval_seconds,
                }
            )

            continue

        retrieval_seconds = (
            time.perf_counter()
            - retrieval_start
        )

        retrieval_nodes = (
            current_step.retrieval_nodes
            or []
        )

        ids = node_ids_from_dicts(
            retrieval_nodes
        )

        print()
        print("-" * 90)
        print("BASELINE RETRIEVAL")
        print("-" * 90)

        print(
            "Tree IDs:",
            ids,
        )

        print(
            "Evidence count:",
            len(retrieval_nodes),
        )

        print(
            "Retrieval time:",
            round(
                retrieval_seconds,
                3,
            ),
            "seconds",
        )

        # --------------------------------------------------------------
        # EAM assessment.
        # --------------------------------------------------------------

        assessment = eam.assess(
            query=query,
            retrieval_nodes=retrieval_nodes,
        )

        print()
        print("-" * 90)
        print("EAM")
        print("-" * 90)

        print(
            "Status:",
            enum_value(
                assessment.status
            ),
        )

        print(
            "Diagnosis:",
            enum_value(
                assessment.diagnosis
            ),
        )

        print(
            "Relevance:",
            round(
                assessment.relevance_score,
                3,
            ),
        )

        print(
            "Coverage:",
            round(
                assessment.coverage_score,
                3,
            ),
        )

        print(
            "Support:",
            round(
                assessment.support_score,
                3,
            ),
        )

        print(
            "Contradiction:",
            round(
                assessment.contradiction_score,
                3,
            ),
        )

        print(
            "Progress:",
            round(
                assessment.progress_score,
                3,
            ),
        )

        # --------------------------------------------------------------
        # Controller decision.
        # --------------------------------------------------------------

        state = RetryState()

        decision = controller.decide(
            assessment,
            state,
        )

        strategy = enum_value(
            decision.strategy
        )

        action = enum_value(
            decision.action
        )

        stop_reason = enum_value(
            decision.stop_reason
        )

        print()
        print("-" * 90)
        print("CONTROLLER")
        print("-" * 90)

        print(
            "Action:",
            action,
        )

        print(
            "Strategy:",
            strategy,
        )

        print(
            "Stop reason:",
            stop_reason,
        )

        # --------------------------------------------------------------
        # Screening verdict.
        # --------------------------------------------------------------

        promising = (
            decision.strategy
            is not None
        )

        print()
        print("-" * 90)
        print("VERDICT")
        print("-" * 90)

        if promising:

            print(
                "*** PROMISING RECOVERY CANDIDATE ***"
            )

            print(
                "Suggested strategy:",
                strategy,
            )

        else:

            print(
                "No recovery triggered."
            )

        result = {
            "name": candidate["name"],
            "query": query,
            "expected": candidate["expected"],
            "retrieved_node_ids": ids,
            "evidence_count": len(
                retrieval_nodes
            ),
            "retrieval_seconds": round(
                retrieval_seconds,
                3,
            ),
            "assessment": {
                "status": enum_value(
                    assessment.status
                ),
                "diagnosis": enum_value(
                    assessment.diagnosis
                ),
                "relevance_score": (
                    assessment.relevance_score
                ),
                "coverage_score": (
                    assessment.coverage_score
                ),
                "support_score": (
                    assessment.support_score
                ),
                "contradiction_score": (
                    assessment.contradiction_score
                ),
                "progress_score": (
                    assessment.progress_score
                ),
                "missing_entities": (
                    assessment.missing_entities
                ),
                "missing_content": (
                    assessment.missing_content
                ),
                "rationale": (
                    assessment.rationale
                ),
            },
            "decision": {
                "action": action,
                "strategy": strategy,
                "stop_reason": stop_reason,
            },
            "promising_recovery_candidate": (
                promising
            ),
        }

        results.append(
            result
        )

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------

    print()
    print()
    print("=" * 90)
    print("FINAL SCREENING SUMMARY")
    print("=" * 90)

    promising_results = [
        result
        for result in results
        if result.get(
            "promising_recovery_candidate"
        )
    ]

    print()
    print(
        "Total candidates:",
        len(results),
    )

    print(
        "Promising recovery candidates:",
        len(promising_results),
    )

    if promising_results:

        print()

        for result in promising_results:

            print(
                "*",
                result["name"],
            )

            print(
                "  Status:",
                result["assessment"]["status"],
            )

            print(
                "  Diagnosis:",
                result["assessment"]["diagnosis"],
            )

            print(
                "  Strategy:",
                result["decision"]["strategy"],
            )

            print(
                "  IDs:",
                result["retrieved_node_ids"],
            )

            print(
                "  Query:",
                result["query"],
            )

            print()

    else:

        print()
        print(
            "No candidate triggered recovery."
        )

        print(
            "Do NOT run full SC comparisons yet."
        )

    # ------------------------------------------------------------------
    # 5. Save artifact.
    # ------------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Saved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()