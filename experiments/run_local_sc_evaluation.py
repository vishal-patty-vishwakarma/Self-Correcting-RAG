import sys
from pathlib import Path

# Add the BookRAG project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import json
from datetime import datetime
from typing import Any, Dict, List

from Core.Index.Tree import DocumentTree

from Core.rag.self_correcting import (
    ClosedLoopOrchestrator,
    EvidenceAssessmentModule,
    EvidenceDiagnosis,
    EvidenceStatus,
    RecoveryExecutor,
    RecoveryStrategy,
    RetryController,
    RetryState,
)

# Workspace root that holds this repository (…/BookRAG/BookRAG).
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

TREE_PATH = str(
    WORKSPACE_ROOT / "test_output" / "controlled_manual" / "tree.pkl"
)

OUTPUT_DIR = (
    WORKSPACE_ROOT / "test_output" / "controlled_manual" / "evaluation"
)


def enum_value(value):
    if value is None:
        return None

    return getattr(
        value,
        "value",
        value,
    )


def evidence_ids(nodes: List[Any]) -> List[str]:
    result = []

    for node in nodes:

        if isinstance(node, dict):
            node_id = node.get("index_id")
        else:
            node_id = getattr(
                node,
                "index_id",
                None,
            )

        if node_id is not None:
            result.append(str(node_id))

    return result


def make_record(
    case_id: str,
    query: str,
    expected_status: str,
    expected_diagnosis: str,
) -> Dict[str, Any]:

    return {
        "case_id": case_id,
        "query": query,

        "expected_status": expected_status,
        "expected_diagnosis": expected_diagnosis,

        "initial_status": None,
        "initial_diagnosis": None,
        "initial_relevance": None,
        "initial_coverage": None,
        "initial_support": None,

        "final_status": None,
        "final_diagnosis": None,
        "final_relevance": None,
        "final_coverage": None,
        "final_support": None,
        "contradiction_score": None,

        "attempts": 0,
        "strategies": [],
        "stop_reason": None,

        "initial_evidence_ids": [],
        "final_evidence_ids": [],

        "passed": False,
        "notes": "",
    }


def run_orchestrated_case(
    case_id: str,
    query: str,
    initial_nodes: List[Any],
    expected_status: EvidenceStatus,
    expected_diagnosis: EvidenceDiagnosis,
    orchestrator: ClosedLoopOrchestrator,
    eam: EvidenceAssessmentModule,
) -> Dict[str, Any]:

    record = make_record(
        case_id,
        query,
        expected_status.value,
        expected_diagnosis.value,
    )

    # --------------------------------------------------------------
    # Initial evidence assessment
    # --------------------------------------------------------------

    initial_assessment = eam.assess(
        query=query,
        retrieval_nodes=initial_nodes,
    )

    record["initial_status"] = (
        initial_assessment.status.value
    )

    record["initial_diagnosis"] = (
        initial_assessment.diagnosis.value
    )

    record["initial_relevance"] = (
        initial_assessment.relevance_score
    )

    record["initial_coverage"] = (
        initial_assessment.coverage_score
    )

    record["initial_support"] = (
        initial_assessment.support_score
    )

    record["initial_evidence_ids"] = (
        initial_assessment.evidence_ids
    )

    # --------------------------------------------------------------
    # Closed-loop execution
    # --------------------------------------------------------------

    result = orchestrator.run(
        query=query,
        initial_nodes=initial_nodes,
    )

    final = result.final_assessment

    record["final_status"] = final.status.value
    record["final_diagnosis"] = (
        final.diagnosis.value
    )

    record["final_relevance"] = (
        final.relevance_score
    )

    record["final_coverage"] = (
        final.coverage_score
    )

    record["final_support"] = (
        final.support_score
    )

    record["contradiction_score"] = (
        final.contradiction_score
    )

    record["attempts"] = (
        result.retry_state.attempt_count
    )

    record["strategies"] = [
        strategy.value
        for strategy
        in result.retry_state.strategy_history
    ]

    record["final_evidence_ids"] = (
        final.evidence_ids
    )

    if result.decision_history:

        last_decision = (
            result.decision_history[-1]
        )

        if last_decision.stop_reason:
            record["stop_reason"] = (
                last_decision.stop_reason.value
            )

    record["passed"] = (
        final.status == expected_status
        and final.diagnosis
        == expected_diagnosis
    )

    return record


def run_eam_only_case(
    case_id: str,
    query: str,
    nodes: List[Any],
    expected_status: EvidenceStatus,
    expected_diagnosis: EvidenceDiagnosis,
    eam: EvidenceAssessmentModule,
    controller: RetryController,
) -> Dict[str, Any]:

    record = make_record(
        case_id,
        query,
        expected_status.value,
        expected_diagnosis.value,
    )

    assessment = eam.assess(
        query=query,
        retrieval_nodes=nodes,
    )

    decision = controller.decide(
        assessment=assessment,
        retry_state=RetryState(),
    )

    record["initial_status"] = (
        assessment.status.value
    )

    record["initial_diagnosis"] = (
        assessment.diagnosis.value
    )

    record["initial_relevance"] = (
        assessment.relevance_score
    )

    record["initial_coverage"] = (
        assessment.coverage_score
    )

    record["initial_support"] = (
        assessment.support_score
    )

    record["final_status"] = (
        assessment.status.value
    )

    record["final_diagnosis"] = (
        assessment.diagnosis.value
    )

    record["final_relevance"] = (
        assessment.relevance_score
    )

    record["final_coverage"] = (
        assessment.coverage_score
    )

    record["final_support"] = (
        assessment.support_score
    )

    record["contradiction_score"] = (
        assessment.contradiction_score
    )

    record["initial_evidence_ids"] = (
        assessment.evidence_ids
    )

    record["final_evidence_ids"] = (
        assessment.evidence_ids
    )

    if decision.strategy:
        record["strategies"] = [
            decision.strategy.value
        ]

    if decision.stop_reason:
        record["stop_reason"] = (
            decision.stop_reason.value
        )

    record["passed"] = (
        assessment.status == expected_status
        and assessment.diagnosis
        == expected_diagnosis
    )

    return record


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tree = DocumentTree.load_from_file(
        TREE_PATH
    )

    eam = EvidenceAssessmentModule()

    controller = RetryController()

    recovery = RecoveryExecutor(
        tree=tree,
        max_nodes=25,
    )

    orchestrator = ClosedLoopOrchestrator(
        eam=eam,
        controller=controller,
        recovery_executor=recovery,
    )

    results: List[Dict[str, Any]] = []

    # ==============================================================
    # CASE 1 — DIRECT FACT
    # ==============================================================

    results.append(
        run_orchestrated_case(
            case_id="direct_fact",
            query="Who manufactures Valve X?",
            initial_nodes=[
                {
                    "index_id": 13,
                    "text": (
                        "Valve X is manufactured by "
                        "Asterion Controls Ltd."
                    ),
                }
            ],
            expected_status=(
                EvidenceStatus.SUFFICIENT
            ),
            expected_diagnosis=(
                EvidenceDiagnosis.NONE
            ),
            orchestrator=orchestrator,
            eam=eam,
        )
    )

    # ==============================================================
    # CASE 2 — MULTI-HOP / RECOVERY
    # ==============================================================

    results.append(
        run_orchestrated_case(
            case_id="multihop_recovery",
            query=(
                "What strainer protects the pump "
                "that serves Reactor R-2?"
            ),
            initial_nodes=[
                {
                    "index_id": 37,
                    "text": (
                        "Reactor R-2 is served by "
                        "Cooling Loop C."
                    ),
                }
            ],
            expected_status=(
                EvidenceStatus.SUFFICIENT
            ),
            expected_diagnosis=(
                EvidenceDiagnosis.NONE
            ),
            orchestrator=orchestrator,
            eam=eam,
        )
    )

    # ==============================================================
    # CASE 3 — CONTROLLED ABSENCE
    # ==============================================================

    results.append(
        run_orchestrated_case(
            case_id="controlled_absence",
            query=(
                "What is the operating voltage "
                "of Valve X?"
            ),
            initial_nodes=[
                tree.get_node_by_index_id(44),
                tree.get_node_by_index_id(45),
            ],
            expected_status=(
                EvidenceStatus.SUFFICIENT
            ),
            expected_diagnosis=(
                EvidenceDiagnosis.NONE
            ),
            orchestrator=orchestrator,
            eam=eam,
        )
    )

    # ==============================================================
    # CASE 4 — RESOLVED SUPERSESSION
    # ==============================================================

    results.append(
        run_orchestrated_case(
            case_id="resolved_supersession",
            query=(
                "What is the current inspection "
                "interval for Valve X?"
            ),
            initial_nodes=[
                tree.get_node_by_index_id(27),
                tree.get_node_by_index_id(43),
            ],
            expected_status=(
                EvidenceStatus.SUFFICIENT
            ),
            expected_diagnosis=(
                EvidenceDiagnosis.NONE
            ),
            orchestrator=orchestrator,
            eam=eam,
        )
    )

    # ==============================================================
    # CASE 5 — UNRESOLVED CONTRADICTION
    # ==============================================================

    results.append(
        run_eam_only_case(
            case_id="unresolved_contradiction",
            query=(
                "What is the inspection interval "
                "for Valve X?"
            ),
            nodes=[
                {
                    "index_id": 900,
                    "text": (
                        "Valve X inspection interval "
                        "is 450 hours."
                    ),
                },
                {
                    "index_id": 901,
                    "text": (
                        "Valve X inspection interval "
                        "is 500 hours."
                    ),
                },
            ],
            expected_status=(
                EvidenceStatus.CONTRADICTORY
            ),
            expected_diagnosis=(
                EvidenceDiagnosis.CONTRADICTION
            ),
            eam=eam,
            controller=controller,
        )
    )

    # ==============================================================
    # CASE 6 — NO EVIDENCE
    # ==============================================================

    results.append(
        run_eam_only_case(
            case_id="no_evidence",
            query="Who manufactures Valve X?",
            nodes=[],
            expected_status=(
                EvidenceStatus.INSUFFICIENT
            ),
            expected_diagnosis=(
                EvidenceDiagnosis.LOW_RELEVANCE
            ),
            eam=eam,
            controller=controller,
        )
    )

    # ==============================================================
    # OUTPUT
    # ==============================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    json_path = OUTPUT_DIR / (
        f"local_sc_evaluation_{timestamp}.json"
    )

    csv_path = OUTPUT_DIR / (
        f"local_sc_evaluation_{timestamp}.csv"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    fieldnames = list(
        results[0].keys()
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in results:

            row = dict(row)

            row["strategies"] = (
                ";".join(row["strategies"])
            )

            row["initial_evidence_ids"] = (
                ";".join(
                    row["initial_evidence_ids"]
                )
            )

            row["final_evidence_ids"] = (
                ";".join(
                    row["final_evidence_ids"]
                )
            )

            writer.writerow(row)

    # ==============================================================
    # CONSOLE SUMMARY
    # ==============================================================

    print()
    print("=" * 80)
    print("SELF-CORRECTING RAG LOCAL EVALUATION")
    print("=" * 80)

    for row in results:

        status = (
            "PASS"
            if row["passed"]
            else "FAIL"
        )

        print(
            f"{status:4} | "
            f"{row['case_id']:28} | "
            f"{row['initial_status']:14} -> "
            f"{row['final_status']:14} | "
            f"attempts={row['attempts']} | "
            f"strategies={row['strategies']}"
        )

    passed = sum(
        1
        for row in results
        if row["passed"]
    )

    print("-" * 80)

    print(
        f"Passed: {passed}/{len(results)}"
    )

    print(
        f"JSON: {json_path}"
    )

    print(
        f"CSV : {csv_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()