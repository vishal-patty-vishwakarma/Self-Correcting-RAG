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

# Exact same controlled starting evidence for BOTH branches.
INITIAL_NODE_IDS = [38]

EXPECTED_IDENTIFIER = "ST-C40"


def find_typed_config(obj, target_type, max_depth=5):
    """
    Find a config object inside the loaded SystemConfig
    without assuming its exact field name.
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


def normalize_tree_data(tree_data):
    """
    AnswerAgent increments page by 1 in-place.

    Some structural nodes can have page=None,
    so normalize those before answer generation.
    """

    for node_data in tree_data:
        if node_data.get("page") is None:
            node_data["page"] = 0

    return tree_data


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


def contains_identifier(answer):
    return (
        EXPECTED_IDENTIFIER.upper()
        in answer.upper()
    )


def main():
    print("=" * 88)
    print("FIXED-EVIDENCE BASELINE VS REAL-S3 COMPARISON")
    print("=" * 88)

    print()
    print("Query:", QUERY)
    print("Expected identifier:", EXPECTED_IDENTIFIER)
    print("Controlled initial evidence:", INITIAL_NODE_IDS)

    # ==============================================================
    # 1. Load the same real BookRAG indexes/configuration.
    # ==============================================================

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

    # Use BookRAG's normal factory.
    rag = create_rag_agent(
        strategy_config=rag_cfg,
        llm_config=llm_cfg,
        vlm_config=vlm_cfg,
        gbc_index=gbc,
    )

    print()
    print(
        "Self-correcting infrastructure enabled:",
        rag.self_correcting_enabled,
    )

    # ==============================================================
    # 2. BASELINE BRANCH
    #
    # Fixed node [38] goes directly to the normal AnswerAgent.
    #
    # No EAM.
    # No Retry Controller.
    # No recovery.
    #
    # This is a FIXED-EVIDENCE baseline, not a normal stochastic
    # BookRAG retrieval baseline.
    # ==============================================================

    baseline_nodes = gbc.TreeIndex.get_nodes_data(
        INITIAL_NODE_IDS
    )

    if not baseline_nodes:
        raise RuntimeError(
            "No baseline Tree data returned."
        )

    baseline_nodes = normalize_tree_data(
        baseline_nodes
    )

    print()
    print("=" * 88)
    print("BASELINE — FIXED EVIDENCE ONLY")
    print("=" * 88)

    print(
        "Evidence IDs:",
        node_ids_from_dicts(baseline_nodes),
    )

    baseline_start = time.perf_counter()

    baseline_answer, baseline_partial_answers = (
        rag.answer.answer_simple_question(
            query=QUERY,
            retrieved_nodes=baseline_nodes,
            entities=None,
        )
    )

    baseline_seconds = (
        time.perf_counter()
        - baseline_start
    )

    baseline_identifier_present = (
        contains_identifier(baseline_answer)
    )

    print()
    print("BASELINE ANSWER:")
    print(baseline_answer)

    print()
    print(
        "Contains ST-C40:",
        baseline_identifier_present,
    )

    print(
        "Baseline answer time:",
        round(baseline_seconds, 3),
        "seconds",
    )

    # ==============================================================
    # 3. SELF-CORRECTING BRANCH
    #
    # Start from a fresh copy of EXACTLY the same Tree evidence.
    #
    # [38]
    #   -> EAM
    #   -> Retry Controller
    #   -> S3
    #   -> real KG
    #   -> reassessment
    #   -> BookRAG AnswerAgent
    # ==============================================================

    sc_initial_nodes = gbc.TreeIndex.get_nodes_data(
        INITIAL_NODE_IDS
    )

    if not sc_initial_nodes:
        raise RuntimeError(
            "No SC initial Tree data returned."
        )

    sc_initial_nodes = normalize_tree_data(
        sc_initial_nodes
    )

    current_step = SubStep(
        sub_query=QUERY,
        sub_number=1,
    )

    rag._process_retrieved_nodes(
        sc_initial_nodes,
        current_step,
    )

    initial_assessment = rag.sc_eam.assess(
        query=QUERY,
        retrieval_nodes=current_step.retrieval_nodes,
    )

    print()
    print("=" * 88)
    print("SELF-CORRECTING — INITIAL STATE")
    print("=" * 88)

    print(
        "Initial evidence IDs:",
        node_ids_from_dicts(
            current_step.retrieval_nodes
        ),
    )

    print(
        "Initial status:",
        initial_assessment.status.value,
    )

    print(
        "Initial diagnosis:",
        initial_assessment.diagnosis.value,
    )

    print(
        "Initial coverage:",
        initial_assessment.coverage_score,
    )

    print(
        "Initial support:",
        initial_assessment.support_score,
    )

    # --------------------------------------------------------------
    # Execute the REAL live integration method.
    # --------------------------------------------------------------

    recovery_start = time.perf_counter()

    rag._apply_self_correcting_recovery(
        query=QUERY,
        current_step=current_step,
    )

    recovery_seconds = (
        time.perf_counter()
        - recovery_start
    )

    sc_final_ids = node_ids_from_dicts(
        current_step.retrieval_nodes
    )

    print()
    print("=" * 88)
    print("SELF-CORRECTING — RECOVERY RESULT")
    print("=" * 88)

    print(
        "Final status:",
        current_step.self_correcting_final_status,
    )

    print(
        "Stop reason:",
        current_step.self_correcting_stop_reason,
    )

    print(
        "Final evidence IDs:",
        sc_final_ids,
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

    # --------------------------------------------------------------
    # Answer using the recovered evidence.
    # --------------------------------------------------------------

    sc_answer_start = time.perf_counter()

    sc_answer, sc_partial_answers = (
        rag.answer.answer_simple_question(
            query=QUERY,
            retrieved_nodes=current_step.retrieval_nodes,
            entities=None,
        )
    )

    sc_answer_seconds = (
        time.perf_counter()
        - sc_answer_start
    )

    sc_total_seconds = (
        recovery_seconds
        + sc_answer_seconds
    )

    sc_identifier_present = (
        contains_identifier(sc_answer)
    )

    print()
    print("=" * 88)
    print("SELF-CORRECTING — FINAL ANSWER")
    print("=" * 88)

    print()
    print(sc_answer)

    print()
    print(
        "Contains ST-C40:",
        sc_identifier_present,
    )

    print(
        "SC answer-generation time:",
        round(sc_answer_seconds, 3),
        "seconds",
    )

    print(
        "SC recovery + answer time:",
        round(sc_total_seconds, 3),
        "seconds",
    )

    # ==============================================================
    # 4. PAIRED COMPARISON
    # ==============================================================

    added_ids = [
        node_id
        for node_id in sc_final_ids
        if node_id not in INITIAL_NODE_IDS
    ]

    print()
    print("=" * 88)
    print("PAIRED COMPARISON")
    print("=" * 88)

    print()
    print("Same query:              YES")
    print("Same initial evidence:   YES")
    print(
        "Initial evidence IDs:   ",
        INITIAL_NODE_IDS,
    )

    print()
    print(
        "Baseline final IDs:     ",
        INITIAL_NODE_IDS,
    )

    print(
        "SC final IDs:           ",
        sc_final_ids,
    )

    print(
        "Nodes added by recovery:",
        added_ids,
    )

    print()
    print(
        "Baseline contains ST-C40:",
        baseline_identifier_present,
    )

    print(
        "SC contains ST-C40:      ",
        sc_identifier_present,
    )

    print()
    print(
        "Baseline answer time:",
        round(baseline_seconds, 3),
        "seconds",
    )

    print(
        "SC recovery time:",
        round(recovery_seconds, 3),
        "seconds",
    )

    print(
        "SC answer time:",
        round(sc_answer_seconds, 3),
        "seconds",
    )

    print(
        "SC total time:",
        round(sc_total_seconds, 3),
        "seconds",
    )

    # ==============================================================
    # 5. Controlled interpretation
    # ==============================================================

    if (
        not baseline_identifier_present
        and sc_identifier_present
    ):
        outcome = (
            "FIXED_EVIDENCE_RECOVERY_IMPROVEMENT"
        )

        interpretation = (
            "With identical fixed initial evidence, "
            "the baseline answer omitted the expected "
            "identifier while S3 recovery produced evidence "
            "that allowed BookRAG to generate ST-C40."
        )

    elif (
        baseline_identifier_present
        and sc_identifier_present
    ):
        outcome = (
            "BOTH_FIXED_EVIDENCE_ANSWERS_CORRECT"
        )

        interpretation = (
            "The fixed initial evidence was already sufficient "
            "for answer generation. S3 expanded the evidence, "
            "but this case does not demonstrate an answer-accuracy gain."
        )

    elif (
        baseline_identifier_present
        and not sc_identifier_present
    ):
        outcome = (
            "FIXED_EVIDENCE_SC_DEGRADATION"
        )

        interpretation = (
            "The baseline contained the expected identifier "
            "but the recovered-evidence answer did not."
        )

    else:
        outcome = (
            "NO_FIXED_EVIDENCE_ANSWER_IMPROVEMENT"
        )

        interpretation = (
            "Neither branch produced the expected identifier."
        )

    print()
    print("=" * 88)
    print("CONTROLLED OUTCOME")
    print("=" * 88)

    print()
    print("Outcome:", outcome)

    print()
    print(interpretation)

    # ==============================================================
    # 6. Save research artifact
    # ==============================================================

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
        / "fixed_baseline_vs_real_s3_node38.json"
    )

    result_record = {
        "experiment": (
            "fixed_evidence_baseline_vs_real_s3"
        ),
        "experiment_type": (
            "deterministic_fixed_evidence_comparison"
        ),
        "config": CONFIG,
        "query": QUERY,
        "expected_identifier": EXPECTED_IDENTIFIER,
        "initial_tree_ids": INITIAL_NODE_IDS,

        "baseline": {
            "tree_ids": INITIAL_NODE_IDS,
            "answer": baseline_answer,
            "expected_identifier_present": (
                baseline_identifier_present
            ),
            "answer_seconds": baseline_seconds,
        },

        "self_correcting": {
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
            "final_tree_ids": sc_final_ids,
            "added_tree_ids": added_ids,
            "assessments": (
                current_step.self_correcting_assessments
            ),
            "decisions": (
                current_step.self_correcting_decisions
            ),
            "answer": sc_answer,
            "expected_identifier_present": (
                sc_identifier_present
            ),
            "recovery_seconds": recovery_seconds,
            "answer_seconds": sc_answer_seconds,
            "total_seconds": sc_total_seconds,
        },

        "outcome": outcome,
        "interpretation": interpretation,
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
        "Saved comparison artifact:",
        output_path,
    )

    print()
    print("=" * 88)
    print("EXPERIMENT COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()