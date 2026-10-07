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


CONFIG = "config/sc_stress.yaml"

QUERY = (
    "What strainer protects the pump "
    "that serves Reactor R-2?"
)

# Deterministic weak real evidence.
#
# This node was already runtime-validated to produce:
#
# PARTIAL / MULTI_HOP_GAP
# -> S3_GRAPH_MULTI_HOP
# -> SUFFICIENT
#
INITIAL_NODE_IDS = [38]

EXPECTED_IDENTIFIER = "ST-C40"


def find_typed_config(obj, target_type, max_depth=5):
    """
    Find an instantiated config object inside the loaded
    SystemConfig without assuming the exact top-level field name.

    This allows us to reuse BookRAG's real create_rag_agent()
    factory instead of manually reconstructing the LLM/VLM/RAG.
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
    print("FIXED-EVIDENCE REAL S3 + BOOKRAG ANSWER REPLAY")
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
        "S3 available:",
        rag.sc_recovery.is_available(
            __import__(
                "Core.rag.self_correcting",
                fromlist=["RecoveryStrategy"],
            ).RecoveryStrategy.S3_GRAPH_MULTI_HOP
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

    print(
        "Initial retrieval IDs:",
        node_ids_from_dicts(
            current_step.retrieval_nodes
        ),
    )

    initial_assessment = rag.sc_eam.assess(
        query=QUERY,
        retrieval_nodes=current_step.retrieval_nodes,
    )

    print(
        "Status:",
        initial_assessment.status.value,
    )

    print(
        "Diagnosis:",
        initial_assessment.diagnosis.value,
    )

    print(
        "Coverage:",
        initial_assessment.coverage_score,
    )

    print(
        "Support:",
        initial_assessment.support_score,
    )

    # ------------------------------------------------------------------
    # 4. Execute the ACTUAL live BookRAG self-correcting integration.
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
            f"coverage={assessment.get('coverage_score')} | "
            f"support={assessment.get('support_score')}"
        )

    print()
    print("DECISION TRACE")

    for i, decision in enumerate(
        current_step.self_correcting_decisions
    ):
        print(
            f"Decision {i}: "
            f"strategy={decision.get('strategy')} | "
            f"stop={decision.get('stop_reason')}"
        )

    # ------------------------------------------------------------------
    # 5. Pass the recovered evidence into BookRAG's REAL AnswerAgent.
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

    identifier_recovered = (
        EXPECTED_IDENTIFIER.upper()
        in final_answer.upper()
    )

    print()
    print("FINAL ANSWER:")
    print(final_answer)

    print()
    print(
        "Expected identifier:",
        EXPECTED_IDENTIFIER,
    )

    print(
        "Exact identifier present:",
        identifier_recovered,
    )

    print(
        "Answer-generation time:",
        round(answer_seconds, 3),
        "seconds",
    )

    print(
        "Total experiment time:",
        round(
            recovery_seconds
            + answer_seconds,
            3,
        ),
        "seconds",
    )

    # ------------------------------------------------------------------
    # 6. Save a reproducible research artifact.
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
        / "real_s3_with_answer_node38.json"
    )

    result_record = {
        "experiment": (
            "fixed_evidence_real_s3_with_bookrag_answer"
        ),
        "config": CONFIG,
        "query": QUERY,
        "expected_identifier": EXPECTED_IDENTIFIER,
        "initial_tree_ids": INITIAL_NODE_IDS,
        "initial_status": (
            initial_assessment.status.value
        ),
        "initial_diagnosis": (
            initial_assessment.diagnosis.value
        ),
        "initial_coverage": (
            initial_assessment.coverage_score
        ),
        "initial_support": (
            initial_assessment.support_score
        ),
        "final_status": (
            current_step.self_correcting_final_status
        ),
        "stop_reason": (
            current_step.self_correcting_stop_reason
        ),
        "final_tree_ids": final_ids,
        "self_correcting_assessments": (
            current_step.self_correcting_assessments
        ),
        "self_correcting_decisions": (
            current_step.self_correcting_decisions
        ),
        "final_answer": final_answer,
        "expected_identifier_present": (
            identifier_recovered
        ),
        "recovery_seconds": recovery_seconds,
        "answer_generation_seconds": (
            answer_seconds
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

    if identifier_recovered:
        print(
            "RESULT: RECOVERED EVIDENCE PRODUCED "
            "THE EXPECTED IDENTIFIER."
        )
    else:
        print(
            "RESULT: RECOVERY COMPLETED, BUT THE "
            "GENERATED ANSWER DID NOT CONTAIN "
            "THE EXPECTED IDENTIFIER."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()