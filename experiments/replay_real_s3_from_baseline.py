import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Core.configs.system_config import load_system_config
from Core.Index.GBCIndex import GBC

from Core.rag.self_correcting import (
    ClosedLoopOrchestrator,
    EvidenceAssessmentModule,
    RecoveryExecutor,
    RetryController,
    RetryState,
)


CONFIG = "config/sc_stress.yaml"

QUERY = (
    "What strainer protects the pump "
    "that serves Reactor R-2?"
)

# Exact weak evidence returned by the real
# baseline topk=1 experiment.
BASELINE_NODE_IDS = [38]


def main():

    print("=" * 80)
    print("FIXED-EVIDENCE REAL S3 REPLAY")
    print("=" * 80)

    cfg = load_system_config(CONFIG)

    gbc = GBC.load_gbc_index(cfg)

    initial_nodes = [
        gbc.TreeIndex.get_node_by_index_id(node_id)
        for node_id in BASELINE_NODE_IDS
    ]

    initial_nodes = [
        node
        for node in initial_nodes
        if node is not None
    ]

    eam = EvidenceAssessmentModule()

    controller = RetryController()

    recovery = RecoveryExecutor(
        tree=gbc.TreeIndex,
        graph_index=gbc.GraphIndex,
        max_nodes=25,
        graph_hops=2,
    )

    orchestrator = ClosedLoopOrchestrator(
        eam=eam,
        controller=controller,
        recovery_executor=recovery,
    )

    print()
    print("Initial Tree IDs:", BASELINE_NODE_IDS)

    initial_assessment = eam.assess(
        query=QUERY,
        retrieval_nodes=initial_nodes,
    )

    print()
    print("INITIAL ASSESSMENT")
    print("-" * 80)

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

    print()
    print(
        "S3 available:",
        recovery.is_available(
            recovery_strategy := __import__(
                "Core.rag.self_correcting",
                fromlist=["RecoveryStrategy"],
            ).RecoveryStrategy.S3_GRAPH_MULTI_HOP
        ),
    )

    result = orchestrator.run(
        query=QUERY,
        initial_nodes=initial_nodes,
        retry_state=RetryState(
            max_recovery_attempts=4,
        ),
    )

    print()
    print("=" * 80)
    print("RECOVERY TRACE")
    print("=" * 80)

    for i, assessment in enumerate(
        result.assessment_history
    ):

        print(
            f"Assessment {i}: "
            f"{assessment.status.value} | "
            f"{assessment.diagnosis.value} | "
            f"coverage={assessment.coverage_score:.2f} | "
            f"support={assessment.support_score:.2f}"
        )

    print()

    for i, decision in enumerate(
        result.decision_history
    ):

        strategy = (
            decision.strategy.value
            if decision.strategy
            else None
        )

        reason = (
            decision.stop_reason.value
            if decision.stop_reason
            else None
        )

        print(
            f"Decision {i}: "
            f"strategy={strategy} | "
            f"stop={reason}"
        )

    final_ids = []

    for node in result.final_nodes:

        node_id = getattr(
            node,
            "index_id",
            None,
        )

        if node_id is not None:
            final_ids.append(node_id)

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print(
        "Final status:",
        result.final_assessment.status.value,
    )

    print(
        "Final diagnosis:",
        result.final_assessment.diagnosis.value,
    )

    print(
        "Attempts:",
        result.retry_state.attempt_count,
    )

    print(
        "Strategies:",
        [
            strategy.value
            for strategy
            in result.retry_state.strategy_history
        ],
    )

    print(
        "Final Tree IDs:",
        final_ids,
    )


if __name__ == "__main__":
    main()