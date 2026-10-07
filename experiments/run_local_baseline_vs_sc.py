import sys
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Core.Index.Tree import DocumentTree

from Core.rag.self_correcting import (
    ClosedLoopOrchestrator,
    EvidenceAssessmentModule,
    EvidenceDiagnosis,
    EvidenceStatus,
    RecoveryExecutor,
    RetryController,
)


# Workspace root that holds this repository (…/BookRAG/BookRAG).
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

TREE_PATH = str(
    WORKSPACE_ROOT / "test_output" / "controlled_manual" / "tree.pkl"
)

OUTPUT_DIR = (
    WORKSPACE_ROOT / "test_output" / "controlled_manual" / "evaluation"
)


def run_case(
    case_id: str,
    query: str,
    initial_nodes: List[Any],
    ground_truth: str,
    eam: EvidenceAssessmentModule,
    orchestrator: ClosedLoopOrchestrator,
) -> Dict[str, Any]:

    # ==============================================================
    # BASELINE PROXY
    #
    # IMPORTANT:
    # This means the INITIAL evidence assessment before recovery.
    # It is NOT yet the full BookRAG baseline answer.
    # ==============================================================

    initial = eam.assess(
        query=query,
        retrieval_nodes=initial_nodes,
    )

    # ==============================================================
    # SELF-CORRECTING
    # ==============================================================

    result = orchestrator.run(
        query=query,
        initial_nodes=initial_nodes,
    )

    final = result.final_assessment

    strategies = [
        strategy.value
        for strategy in result.retry_state.strategy_history
    ]

    stop_reason = None

    if result.decision_history:
        last = result.decision_history[-1]

        if last.stop_reason:
            stop_reason = last.stop_reason.value

    # ==============================================================
    # IMPROVEMENT SIGNALS
    # ==============================================================

    coverage_gain = (
        final.coverage_score
        - initial.coverage_score
    )

    support_gain = (
        final.support_score
        - initial.support_score
    )

    became_sufficient = (
        initial.status != EvidenceStatus.SUFFICIENT
        and final.status == EvidenceStatus.SUFFICIENT
    )

    recovery_helped = (
        coverage_gain > 0
        or support_gain > 0
        or became_sufficient
    )

    return {
        "case_id": case_id,
        "query": query,
        "ground_truth": ground_truth,

        "baseline_proxy_status": (
            initial.status.value
        ),

        "baseline_proxy_diagnosis": (
            initial.diagnosis.value
        ),

        "baseline_relevance": (
            initial.relevance_score
        ),

        "baseline_coverage": (
            initial.coverage_score
        ),

        "baseline_support": (
            initial.support_score
        ),

        "sc_final_status": (
            final.status.value
        ),

        "sc_final_diagnosis": (
            final.diagnosis.value
        ),

        "sc_relevance": (
            final.relevance_score
        ),

        "sc_coverage": (
            final.coverage_score
        ),

        "sc_support": (
            final.support_score
        ),

        "coverage_gain": coverage_gain,
        "support_gain": support_gain,

        "became_sufficient": became_sufficient,
        "recovery_helped": recovery_helped,

        "recovery_attempts": (
            result.retry_state.attempt_count
        ),

        "strategies_used": strategies,

        "stop_reason": stop_reason,

        "initial_evidence_ids": (
            initial.evidence_ids
        ),

        "final_evidence_ids": (
            final.evidence_ids
        ),
    }


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

    cases = []

    # ==============================================================
    # CASE 1 — DIRECT FACT
    # ==============================================================

    cases.append(
        run_case(
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
            ground_truth="Asterion Controls Ltd.",
            eam=eam,
            orchestrator=orchestrator,
        )
    )

    # ==============================================================
    # CASE 2 — MULTI-HOP RECOVERY
    # ==============================================================

    cases.append(
        run_case(
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
            ground_truth="ST-C40",
            eam=eam,
            orchestrator=orchestrator,
        )
    )

    # ==============================================================
    # CASE 3 — CONTROLLED ABSENCE
    # ==============================================================

    cases.append(
        run_case(
            case_id="controlled_absence",
            query=(
                "What is the operating voltage "
                "of Valve X?"
            ),
            initial_nodes=[
                tree.get_node_by_index_id(44),
                tree.get_node_by_index_id(45),
            ],
            ground_truth=(
                "The manual does not specify "
                "an operating voltage for Valve X."
            ),
            eam=eam,
            orchestrator=orchestrator,
        )
    )

    # ==============================================================
    # CASE 4 — RESOLVED SUPERSESSION
    # ==============================================================

    cases.append(
        run_case(
            case_id="resolved_supersession",
            query=(
                "What is the current inspection "
                "interval for Valve X?"
            ),
            initial_nodes=[
                tree.get_node_by_index_id(27),
                tree.get_node_by_index_id(43),
            ],
            ground_truth="500 operating hours",
            eam=eam,
            orchestrator=orchestrator,
        )
    )

    # ==============================================================
    # SAVE OUTPUT
    # ==============================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    json_path = OUTPUT_DIR / (
        f"local_baseline_vs_sc_{timestamp}.json"
    )

    csv_path = OUTPUT_DIR / (
        f"local_baseline_vs_sc_{timestamp}.csv"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            cases,
            f,
            indent=2,
            ensure_ascii=False,
        )

    fieldnames = list(
        cases[0].keys()
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

        for row in cases:

            row = dict(row)

            row["strategies_used"] = ";".join(
                row["strategies_used"]
            )

            row["initial_evidence_ids"] = ";".join(
                row["initial_evidence_ids"]
            )

            row["final_evidence_ids"] = ";".join(
                row["final_evidence_ids"]
            )

            writer.writerow(row)

    # ==============================================================
    # CONSOLE REPORT
    # ==============================================================

    print()
    print("=" * 100)

    print(
        "LOCAL BASELINE-PROXY VS SELF-CORRECTING RAG"
    )

    print("=" * 100)

    for row in cases:

        print(
            f"{row['case_id']:24} | "
            f"{row['baseline_proxy_status']:13} -> "
            f"{row['sc_final_status']:13} | "
            f"coverage "
            f"{row['baseline_coverage']:.2f} -> "
            f"{row['sc_coverage']:.2f} | "
            f"attempts={row['recovery_attempts']} | "
            f"helped={row['recovery_helped']}"
        )

    recovered = sum(
        1
        for row in cases
        if row["became_sufficient"]
    )

    helped = sum(
        1
        for row in cases
        if row["recovery_helped"]
    )

    print("-" * 100)

    print(
        f"Cases evaluated: {len(cases)}"
    )

    print(
        f"Cases improved by recovery: {helped}"
    )

    print(
        f"Cases converted to SUFFICIENT: {recovered}"
    )

    print()
    print(
        "NOTE: 'baseline' here means initial evidence "
        "before recovery, NOT full BookRAG answer generation."
    )

    print()
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")

    print("=" * 100)


if __name__ == "__main__":
    main()