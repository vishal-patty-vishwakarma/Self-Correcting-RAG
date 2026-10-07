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


CONFIG = "config/self_correcting_kg_fallback.yaml"

QUERY = (
    "For the circulation equipment serving Reactor R-2, "
    "what is its alternate identifier and which inlet "
    "strainer protects it?"
)

# Fixed weak real evidence.
#
# Previous candidate-screening run produced:
#   PARTIAL / MISSING_ENTITY
#   -> expected S2_ENTITY_GUIDED
#
# We now test deterministically with these exact nodes.

INITIAL_NODE_IDS = [34, 38, 39, 40, 17, 18]

EXPECTED_IDENTIFIERS = ["PX-40", "ST-C40"]


def find_typed_config(obj, target_type, max_depth=5):
    """
    Find an instantiated config object inside the loaded
    SystemConfig without assuming the exact top-level field name.
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
            found = walk(child, depth + 1)

            if found is not None:
                return found

        return None

    return walk(obj, 0)


def node_ids_from_dicts(nodes):
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


def main():
    print("=" * 80)
    print("FIXED-EVIDENCE REAL S2 + BOOKRAG ANSWER REPLAY")
    print("=" * 80)

    print()
    print("Config:", CONFIG)
    print("Query:", QUERY)
    print("Initial Tree IDs:", INITIAL_NODE_IDS)

    # ------------------------------------------------------------------
    # 1. Load the exact existing BookRAG system/index.
    # ------------------------------------------------------------------

    system_cfg = load_system_config(CONFIG)

    gbc = GBC.load_gbc_index(system_cfg)

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
            "Could not locate GBCRAGConfig inside the loaded system config."
        )

    if llm_cfg is None:
        raise RuntimeError(
            "Could not locate LLMConfig inside the loaded system config."
        )

    if vlm_cfg is None:
        raise RuntimeError(
            "Could not locate VLMConfig inside the loaded system config."
        )

    print()
    print("Loaded GBCRAGConfig:", type(rag_cfg).__name__)
    print("Loaded LLMConfig:", type(llm_cfg).__name__)
    print("Loaded VLMConfig:", type(vlm_cfg).__name__)

    # ------------------------------------------------------------------
    # 2. Build the normal live BookRAG agent using its own factory.
    # ------------------------------------------------------------------

    rag = create_rag_agent(
        strategy_config=rag_cfg,
        llm_config=llm_cfg,
        vlm_config=vlm_cfg,
        gbc_index=gbc,
    )

    print()
    print(
        "Self-correcting enabled:",
        rag.self_correcting_enabled,
    )

    print(
        "S2 available:",
        rag.sc_recovery.is_available(
            __import__(
                "Core.rag.self_correcting",
                fromlist=["RecoveryStrategy"],
            ).RecoveryStrategy.S2_ENTITY_GUIDED
        ),
    )

    # ------------------------------------------------------------------
    # 3. Create the exact fixed BookRAG evidence representation.
    # ------------------------------------------------------------------

    initial_tree_data = (
        gbc.TreeIndex.get_nodes_data(
            INITIAL_NODE_IDS
        )
    )

    if not initial_tree_data:
        raise RuntimeError(
            "No Tree data was returned for INITIAL_NODE_IDS."
        )

    # Structural nodes may have page=None.
    # AnswerAgent later increments page in-place.
    for node_data in initial_tree_data:
        if node_data.get("page") is None:
            node_data["page"] = 0

    current_step = SubStep(
        sub_query=QUERY,
        sub_number=1,
    )

    rag._process_retrieved_nodes(
        initial_tree_data,
        current_step,
    )

    print()
    print("=" * 80)
    print("INITIAL EVIDENCE")
    print("=" * 80)

    initial_ids = node_ids_from_dicts(
        current_step.retrieval_nodes
    )

    print(
        "Initial retrieval IDs:",
        initial_ids,
    )

    # Print the text content of each initial node
    # for diagnostic traceability.
    for node_data in initial_tree_data:
        nid = node_data.get("index_id")
        text = node_data.get("content", "")
        preview = text[:120].replace("\n", " ")
        print(f"  Node {nid}: {preview}...")

    # ------------------------------------------------------------------
    # 4. Run an initial EAM assessment BEFORE recovery.
    # ------------------------------------------------------------------

    initial_assessment = rag.sc_eam.assess(
        query=QUERY,
        retrieval_nodes=current_step.retrieval_nodes,
    )

    print()
    print("Initial EAM status:",
          initial_assessment.status.value)
    print("Initial diagnosis:",
          initial_assessment.diagnosis.value)
    print("Relevance score:",
          round(initial_assessment.relevance_score, 4))
    print("Coverage score:",
          round(initial_assessment.coverage_score, 4))
    print("Support score:",
          round(initial_assessment.support_score, 4))
    print("Contradiction score:",
          round(initial_assessment.contradiction_score, 4))
    print("Missing entities:",
          initial_assessment.missing_entities)
    print("Rationale:",
          initial_assessment.rationale)

    # ------------------------------------------------------------------
    # 5. Optional baseline: answer using ONLY the fixed initial evidence.
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("BASELINE ANSWER (FIXED INITIAL EVIDENCE ONLY)")
    print("=" * 80)

    baseline_answer = None

    try:
        baseline_answer, _ = (
            rag.answer.answer_simple_question(
                query=QUERY,
                retrieved_nodes=current_step.retrieval_nodes,
                entities=getattr(
                    current_step,
                    "iteration_graph_nodes",
                    None,
                ),
            )
        )

        print("Baseline answer:")
        print(baseline_answer)

        for eid in EXPECTED_IDENTIFIERS:
            present = eid.upper() in baseline_answer.upper()
            print(
                f"  {eid} present in baseline: {present}"
            )

    except Exception as exc:
        print(f"Baseline answer generation failed: {exc}")
        baseline_answer = None

    # ------------------------------------------------------------------
    # 6. Execute the ACTUAL live BookRAG self-correcting integration.
    # ------------------------------------------------------------------

    recovery_start = time.perf_counter()

    rag._apply_self_correcting_recovery(
        query=QUERY,
        current_step=current_step,
    )

    recovery_seconds = (
        time.perf_counter()
        - recovery_start
    )

    final_ids = node_ids_from_dicts(
        current_step.retrieval_nodes
    )

    print()
    print("=" * 80)
    print("SELF-CORRECTING RESULT")
    print("=" * 80)

    print(
        "Final status:",
        current_step.self_correcting_final_status,
    )

    print(
        "Stop reason:",
        current_step.self_correcting_stop_reason,
    )

    print(
        "Final Tree IDs:",
        final_ids,
    )

    print(
        "Recovery time:",
        round(recovery_seconds, 3),
        "seconds",
    )

    print()
    print("ASSESSMENT TRACE")

    for i, assessment in enumerate(
        current_step.self_correcting_assessments
    ):
        print(
            f"Assessment {i}: "
            f"status={assessment.get('status')} | "
            f"diagnosis={assessment.get('diagnosis')} | "
            f"relevance={assessment.get('relevance_score')} | "
            f"coverage={assessment.get('coverage_score')} | "
            f"support={assessment.get('support_score')} | "
            f"contradiction={assessment.get('contradiction_score')}"
        )

    print()
    print("DECISION TRACE")

    for i, decision in enumerate(
        current_step.self_correcting_decisions
    ):
        print(
            f"Decision {i}: "
            f"strategy={decision.get('strategy')} | "
            f"stop={decision.get('stop_reason')} | "
            f"attempt={decision.get('attempt_count')} | "
            f"rationale={decision.get('rationale', '')[:120]}"
        )

    print()
    print("Strategies executed:",
          [d.get("strategy")
           for d in current_step.self_correcting_decisions
           if d.get("strategy")])

    print("Recovery attempt count:",
          current_step.self_correcting_assessments.__len__()
          - 1 if current_step.self_correcting_assessments else 0)

    # ------------------------------------------------------------------
    # 7. Pass the recovered evidence into BookRAG's REAL AnswerAgent.
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("BOOKRAG ANSWER GENERATION")
    print("=" * 80)

    answer_start = time.perf_counter()

    final_answer, partial_answers = (
        rag.answer.answer_simple_question(
            query=QUERY,
            retrieved_nodes=current_step.retrieval_nodes,
            entities=getattr(
                current_step,
                "iteration_graph_nodes",
                None,
            ),
        )
    )

    answer_seconds = (
        time.perf_counter()
        - answer_start
    )

    total_experiment_time = (
        recovery_seconds + answer_seconds
    )

    identifier_checks = {}
    for eid in EXPECTED_IDENTIFIERS:
        identifier_checks[eid] = (
            eid.upper() in final_answer.upper()
        )

    print()
    print("FINAL ANSWER:")
    print(final_answer)

    print()
    for eid in EXPECTED_IDENTIFIERS:
        present = identifier_checks[eid]
        print(
            f"Expected identifier {eid} present: {present}"
        )

    print(
        "Answer-generation time:",
        round(answer_seconds, 3),
        "seconds",
    )

    print(
        "Total experiment time:",
        round(total_experiment_time, 3),
        "seconds",
    )

    # ------------------------------------------------------------------
    # 8. Save a reproducible research artifact.
    # ------------------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "test_output"
        / "controlled_manual"
        / "fixed_evidence_experiments"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "real_s2_with_answer_fixed_evidence_after_eam_fix.json"
    )

    result_record = {
        "experiment": (
            "fixed_evidence_real_s2_with_bookrag_answer"
        ),
        "config": CONFIG,
        "query": QUERY,
        "expected_identifiers": EXPECTED_IDENTIFIERS,
        "initial_node_ids": INITIAL_NODE_IDS,
        "initial_assessment": {
            "status": (
                initial_assessment.status.value
            ),
            "diagnosis": (
                initial_assessment.diagnosis.value
            ),
            "relevance_score": (
                initial_assessment.relevance_score
            ),
            "coverage_score": (
                initial_assessment.coverage_score
            ),
            "support_score": (
                initial_assessment.support_score
            ),
            "contradiction_score": (
                initial_assessment.contradiction_score
            ),
            "missing_entities": (
                initial_assessment.missing_entities
            ),
            "rationale": (
                initial_assessment.rationale
            ),
        },
        "decision_trace": (
            current_step.self_correcting_decisions
        ),
        "assessment_trace": (
            current_step.self_correcting_assessments
        ),
        "strategies": [
            d.get("strategy")
            for d in current_step.self_correcting_decisions
            if d.get("strategy")
        ],
        "attempt_count": len(
            current_step.self_correcting_assessments
        ) - 1 if current_step.self_correcting_assessments else 0,
        "final_node_ids": final_ids,
        "final_status": (
            current_step.self_correcting_final_status
        ),
        "stop_reason": (
            current_step.self_correcting_stop_reason
        ),
        "baseline_answer_if_available": (
            baseline_answer
        ),
        "final_answer": final_answer,
        "expected_identifier_checks": (
            identifier_checks
        ),
        "recovery_seconds": round(
            recovery_seconds, 3
        ),
        "answer_seconds": round(
            answer_seconds, 3
        ),
        "total_seconds": round(
            total_experiment_time, 3
        ),
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result_record,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Saved experiment artifact:",
        output_path,
    )

    print()
    print("=" * 80)

    all_recovered = all(
        identifier_checks.values()
    )

    if all_recovered:
        print(
            "RESULT: RECOVERED EVIDENCE PRODUCED "
            "ALL EXPECTED IDENTIFIERS."
        )
    else:
        missing = [
            eid
            for eid, ok in identifier_checks.items()
            if not ok
        ]
        print(
            "RESULT: RECOVERY COMPLETED, BUT THE "
            "GENERATED ANSWER DID NOT CONTAIN "
            f"ALL EXPECTED IDENTIFIERS. "
            f"Missing: {missing}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
